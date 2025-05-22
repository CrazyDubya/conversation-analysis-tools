import json
import sqlite3
import datetime
import os
import uuid
import sys
from typing import Dict, List, Any, Optional, Union, Tuple, Iterator


class ConversationParser:
    """Universal parser for AI conversation data from different platforms."""

    def __init__(self, db_path: str = "ai_conversations.db"):
        """Initialize the parser with a database path.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.initialize_database()

    def initialize_database(self):
        """Create database tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create conversations table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            title TEXT,
            platform TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP,
            account_id TEXT,
            original_id TEXT,
            metadata TEXT
        )
        """
        )

        # Create messages table
        cursor.execute(
            """
        CREATE TABLE IF NOT EXISTS messages (
            id TEXT PRIMARY KEY,
            conversation_id TEXT,
            parent_id TEXT,
            sender TEXT,
            role TEXT,
            content TEXT,
            created_at TIMESTAMP,
            model TEXT,
            order_index INTEGER,
            metadata TEXT,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id)
        )
        """
        )

        conn.commit()
        conn.close()

    def parse_file(
        self, file_path: str, platform: Optional[str] = None, chunk_size: int = 10
    ):
        """Parse a conversation file and store in database using streaming approach.

        Args:
            file_path: Path to the JSON file
            platform: Override platform detection (optional)
            chunk_size: Number of conversations to process at once
        """
        # Determine file type if not specified
        if not platform:
            platform = self._detect_platform(file_path)
            print(f"Detected platform: {platform}")

        # Process the file based on platform
        if platform == "claude":
            self._process_claude_file(file_path, chunk_size)
        elif platform == "chatgpt":
            self._process_chatgpt_file(file_path, chunk_size)
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def _detect_platform(self, file_path: str) -> str:
        """Detect the platform from file contents.

        Args:
            file_path: Path to the JSON file

        Returns:
            String indicating platform ("claude" or "chatgpt")
        """
        # Read first few bytes to check structure (avoid loading full file)
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_chars = f.read(1000)

            # Check basic structure markers
            if '"chat_messages":' in first_chars:
                return "claude"
            elif '"mapping":' in first_chars:
                return "chatgpt"

            # If not found in first chunk, try to be more thorough
            f.seek(0)
            for i, line in enumerate(f):
                if i > 100:  # Check first 100 lines
                    break
                if '"chat_messages":' in line:
                    return "claude"
                if '"mapping":' in line:
                    return "chatgpt"

        # If we couldn't determine, make a best guess based on filename
        if "claude" in file_path.lower():
            return "claude"
        elif "chatgpt" in file_path.lower() or "gpt" in file_path.lower():
            return "chatgpt"

        raise ValueError("Could not detect platform from file contents")

    def _stream_array_items(self, file_path: str) -> Iterator[str]:
        """Stream JSON array items one by one from a file.

        This function handles the specific case of reading a JSON array
        where each item is a complete JSON object, without loading the entire
        array into memory.

        Args:
            file_path: Path to the JSON file containing an array

        Yields:
            String containing a complete JSON object from the array
        """
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Ensure the file starts with an array
            first_char = f.read(1).strip()
            if first_char != "[":
                raise ValueError(f"File must start with '[', got '{first_char}'")

            # Variables to track parsing state
            buffer = ""
            brace_level = 0
            in_string = False
            escape_next = False

            # Read character by character
            while True:
                char = f.read(1)
                if not char:  # End of file
                    break

                # Update parsing state based on current character
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"' and not escape_next:
                    in_string = not in_string
                elif not in_string:
                    if char == "{":
                        brace_level += 1
                        if brace_level == 1:  # Start of a new object
                            buffer = "{"
                            continue
                    elif char == "}":
                        brace_level -= 1
                        if brace_level == 0:  # End of current object
                            buffer += "}"
                            yield buffer  # Return the complete object
                            buffer = ""

                            # Skip the comma after the object
                            next_char = f.read(1)
                            while next_char and next_char.strip() not in [",", "]"]:
                                next_char = f.read(1)
                            continue

                # Add current character to buffer if we're inside an object
                if brace_level > 0:
                    buffer += char

    def _process_claude_file(self, file_path: str, chunk_size: int):
        """Process Claude conversation data by streaming array items.

        Args:
            file_path: Path to Claude JSON file
            chunk_size: Number of conversations to process at once
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            conversation_count = 0
            for json_str in self._stream_array_items(file_path):
                try:
                    # Parse the conversation JSON
                    conversation = json.loads(json_str)

                    # Process the conversation
                    self._process_claude_conversation(conversation, cursor)
                    conversation_count += 1

                    # Commit periodically
                    if conversation_count % chunk_size == 0:
                        conn.commit()
                        print(f"Processed {conversation_count} Claude conversations...")
                except json.JSONDecodeError as e:
                    print(f"Error parsing Claude conversation: {e}")
                except Exception as e:
                    print(f"Error processing Claude conversation: {e}")

            # Final commit
            conn.commit()
            print(f"Completed processing {conversation_count} Claude conversations.")
        finally:
            conn.close()

    def _process_chatgpt_file(self, file_path: str, chunk_size: int):
        """Process ChatGPT conversation data by streaming array items.

        Args:
            file_path: Path to ChatGPT JSON file
            chunk_size: Number of conversations to process at once
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        try:
            conversation_count = 0
            for json_str in self._stream_array_items(file_path):
                try:
                    # Parse the conversation JSON
                    conversation = json.loads(json_str)

                    # Process the conversation
                    self._process_chatgpt_conversation(conversation, cursor)
                    conversation_count += 1

                    # Commit periodically
                    if conversation_count % chunk_size == 0:
                        conn.commit()
                        print(
                            f"Processed {conversation_count} ChatGPT conversations..."
                        )
                except json.JSONDecodeError as e:
                    print(f"Error parsing ChatGPT conversation: {e}")
                except Exception as e:
                    print(f"Error processing ChatGPT conversation: {e}")

            # Final commit
            conn.commit()
            print(f"Completed processing {conversation_count} ChatGPT conversations.")
        finally:
            conn.close()

    def _process_claude_conversation(
        self, conversation: Dict[str, Any], cursor: sqlite3.Cursor
    ):
        """Process a single Claude conversation.

        Args:
            conversation: Parsed JSON data for one Claude conversation
            cursor: SQLite cursor for database operations
        """
        # Extract conversation metadata
        conv_id = conversation.get("uuid", str(uuid.uuid4()))
        title = conversation.get("name", "")
        created_at = self._parse_timestamp(conversation.get("created_at"))
        updated_at = self._parse_timestamp(conversation.get("updated_at"))
        account_id = None
        if "account" in conversation and isinstance(conversation["account"], dict):
            account_id = conversation["account"].get("uuid", "")

        # Store conversation
        cursor.execute(
            """
        INSERT OR REPLACE INTO conversations 
        (id, title, platform, created_at, updated_at, account_id, original_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                conv_id,
                title,
                "claude",
                created_at,
                updated_at,
                account_id,
                conv_id,
                json.dumps({"uuid": conv_id}),  # Store minimal metadata
            ),
        )

        # Process messages
        chat_messages = conversation.get("chat_messages", [])
        for i, message in enumerate(chat_messages):
            if not isinstance(message, dict):
                continue

            msg_id = message.get("uuid", str(uuid.uuid4()))
            sender = message.get("sender", "")
            content = message.get("text", "")
            msg_created_at = self._parse_timestamp(message.get("created_at"))

            # Extract any potential model information from content
            model = "claude"  # Default
            metadata = {}

            # Check for attachments, files
            if message.get("attachments") or message.get("files"):
                metadata["has_attachments"] = True

            # Store message
            cursor.execute(
                """
            INSERT OR REPLACE INTO messages
            (id, conversation_id, parent_id, sender, role, content, created_at, model, order_index, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    msg_id,
                    conv_id,
                    None,  # Claude doesn't have explicit parent-child relationships
                    sender,
                    sender,  # Role same as sender in Claude
                    content[:100000] if content else "",  # Truncate very long content
                    msg_created_at,
                    model,
                    i,  # Order by position in array
                    json.dumps(metadata),
                ),
            )

    def _process_chatgpt_conversation(
        self, conversation: Dict[str, Any], cursor: sqlite3.Cursor
    ):
        """Process a single ChatGPT conversation without recursion.

        Args:
            conversation: Parsed JSON data for one ChatGPT conversation
            cursor: SQLite cursor for database operations
        """
        # Extract conversation metadata
        conv_id = conversation.get(
            "id", conversation.get("conversation_id", str(uuid.uuid4()))
        )
        title = conversation.get("title", "")
        created_at = self._convert_unix_timestamp(conversation.get("create_time"))
        updated_at = self._convert_unix_timestamp(conversation.get("update_time"))

        # Store conversation
        cursor.execute(
            """
        INSERT OR REPLACE INTO conversations 
        (id, title, platform, created_at, updated_at, account_id, original_id, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                conv_id,
                title,
                "chatgpt",
                created_at,
                updated_at,
                None,  # ChatGPT doesn't have explicit account ID in the data
                conv_id,
                json.dumps({"id": conv_id}),  # Store minimal metadata
            ),
        )

        # Get mapping of messages - FIXED to handle the unhashable slice error
        mapping = conversation.get("mapping", {})
        if not isinstance(mapping, dict):
            print(
                f"Warning: mapping is not a dictionary, skipping message processing for conversation {conv_id}"
            )
            return

        # Iterative approach to traverse the message tree
        message_order = []

        # Find root node (node with no parent)
        root_node = None
        for node_id, node_data in mapping.items():
            # FIXED: Ensure node_id is hashable (a string)
            if not isinstance(node_id, str):
                continue

            if isinstance(node_data, dict) and node_data.get("parent") is None:
                root_node = node_id
                break

        if root_node:
            # Use a stack for iterative traversal (avoids recursion)
            stack = [(root_node, 0)]  # (node_id, depth)
            visited = set()

            while stack:
                node_id, depth = stack.pop(
                    0
                )  # Use as queue for breadth-first traversal

                # FIXED: Ensure node_id is hashable
                if not isinstance(node_id, str):
                    continue

                if node_id in visited:
                    continue

                visited.add(node_id)
                message_order.append(node_id)

                # Add children in reverse order (to process in correct order)
                if node_id in mapping and isinstance(mapping[node_id], dict):
                    children = mapping[node_id].get("children", [])
                    for child_id in reversed(children):
                        # FIXED: Ensure child_id is hashable
                        if isinstance(child_id, str) and child_id not in visited:
                            stack.append((child_id, depth + 1))

        # Process messages in order
        for i, node_id in enumerate(message_order):
            # FIXED: Additional checks for node_id validity
            if not isinstance(node_id, str):
                continue

            if node_id not in mapping or not isinstance(mapping[node_id], dict):
                continue

            node_data = mapping[node_id]
            if not node_data.get("message"):
                continue

            message = node_data.get("message", {})
            if not isinstance(message, dict):
                continue

            parent_id = node_data.get("parent")
            # FIXED: Ensure parent_id is a string
            if not isinstance(parent_id, str) and parent_id is not None:
                parent_id = str(parent_id)

            # Extract author data
            author = message.get("author", {})
            role = author.get("role", "") if isinstance(author, dict) else ""
            sender = (
                "assistant"
                if role == "assistant"
                else "human" if role == "user" else role
            )

            # Extract content
            content_obj = message.get("content", {})
            parts = (
                content_obj.get("parts", []) if isinstance(content_obj, dict) else []
            )

            # FIXED: Handle different types in parts
            content = ""
            if parts and len(parts) > 0:
                if isinstance(parts[0], str):
                    content = parts[0]
                elif isinstance(parts[0], (dict, list)):
                    # Convert non-string parts to JSON string
                    try:
                        content = json.dumps(parts[0])
                    except:
                        content = str(parts[0])

            # Get creation time
            msg_created_at = self._convert_unix_timestamp(message.get("create_time"))

            # Extract model information
            metadata = message.get("metadata", {}) if isinstance(message, dict) else {}
            model = (
                metadata.get("model_slug", "unknown")
                if isinstance(metadata, dict)
                else "unknown"
            )

            # Store message
            try:
                cursor.execute(
                    """
                INSERT OR REPLACE INTO messages
                (id, conversation_id, parent_id, sender, role, content, created_at, model, order_index, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        node_id,
                        conv_id,
                        parent_id,
                        sender,
                        role,
                        (
                            content[:100000] if content else ""
                        ),  # Truncate very long content
                        msg_created_at,
                        model,
                        i,  # Order by traversal position
                        json.dumps({}),  # Store minimal metadata
                    ),
                )
            except Exception as e:
                print(f"Error inserting message: {e}")
                # Continue processing other messages

    def _parse_timestamp(self, timestamp_str: Optional[str]) -> Optional[str]:
        """Parse ISO format timestamp to database format.

        Args:
            timestamp_str: ISO format timestamp string

        Returns:
            Timestamp string in database format
        """
        if not timestamp_str:
            return None

        try:
            dt = datetime.datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return timestamp_str

    def _convert_unix_timestamp(self, timestamp: Optional[float]) -> Optional[str]:
        """Convert Unix timestamp to database format.

        Args:
            timestamp: Unix timestamp

        Returns:
            Timestamp string in database format
        """
        if not timestamp:
            return None

        try:
            dt = datetime.datetime.fromtimestamp(timestamp)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            return None


def main():
    """Main function for running as a command-line script."""
    if len(sys.argv) < 2:
        print("Usage: python fixed_parser.py <json_file1> [<json_file2> ...]")
        print(
            "Example: python fixed_parser.py claude/conversations.json chatgpt/conversations.json"
        )
        return

    parser = ConversationParser("conversations.db")

    for file_path in sys.argv[1:]:
        try:
            print(f"Processing {file_path}...")
            # Set a smaller chunk size for more frequent commits
            parser.parse_file(file_path, chunk_size=10)
            print(f"Successfully processed {file_path}")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            # Print more detailed error info
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    main()
