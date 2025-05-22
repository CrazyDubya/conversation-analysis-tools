-- 1. Human-Assistant Message Pairs
CREATE VIEW IF NOT EXISTS message_pairs AS
WITH numbered_messages AS (
    SELECT 
        conversation_id,
        id,
        sender,
        content,
        created_at,
        ROW_NUMBER() OVER (PARTITION BY conversation_id ORDER BY order_index) AS msg_order
    FROM messages
)
SELECT 
    h.conversation_id,
    h.id AS human_message_id,
    h.content AS human_message,
    h.created_at AS human_timestamp,
    a.id AS assistant_message_id,
    a.content AS assistant_message,
    a.created_at AS assistant_timestamp
FROM numbered_messages h
JOIN numbered_messages a ON 
    h.conversation_id = a.conversation_id AND
    h.msg_order + 1 = a.msg_order
WHERE h.sender = 'human' AND a.sender = 'assistant';

-- 2. Conversation Summary View
CREATE VIEW IF NOT EXISTS conversation_summary AS
SELECT 
    c.id AS conversation_id,
    c.title,
    c.platform,
    c.created_at,
    c.updated_at,
    COUNT(m.id) AS message_count,
    SUM(CASE WHEN m.sender = 'human' THEN 1 ELSE 0 END) AS human_messages,
    SUM(CASE WHEN m.sender = 'assistant' THEN 1 ELSE 0 END) AS assistant_messages,
    MIN(m.created_at) AS first_message_time,
    MAX(m.created_at) AS last_message_time,
    (julianday(MAX(m.created_at)) - julianday(MIN(m.created_at))) * 24 * 60 AS conversation_duration_minutes
FROM conversations c
LEFT JOIN messages m ON c.id = m.conversation_id
GROUP BY c.id;

-- 3. Message Length Statistics
CREATE VIEW IF NOT EXISTS message_length_stats AS
SELECT 
    conversation_id,
    sender,
    COUNT(*) AS message_count,
    AVG(LENGTH(content)) AS avg_length,
    MIN(LENGTH(content)) AS min_length,
    MAX(LENGTH(content)) AS max_length,
    SUM(LENGTH(content)) AS total_length
FROM messages
GROUP BY conversation_id, sender;

-- 4. Time-Based Activity View
CREATE VIEW IF NOT EXISTS time_activity AS
SELECT 
    platform,
    DATE(created_at) AS activity_date,
    COUNT(DISTINCT id) AS conversations,
    COUNT(DISTINCT (
        SELECT MIN(m.id) 
        FROM messages m 
        WHERE m.conversation_id = conversations.id
    )) AS started_conversations
FROM conversations
GROUP BY platform, activity_date
ORDER BY activity_date;

-- 5. Model Usage View
CREATE VIEW IF NOT EXISTS model_usage AS
SELECT 
    c.platform,
    m.model,
    COUNT(*) AS message_count,
    COUNT(DISTINCT m.conversation_id) AS conversation_count,
    AVG(LENGTH(m.content)) AS avg_message_length
FROM messages m
JOIN conversations c ON m.conversation_id = c.id
WHERE m.model IS NOT NULL AND m.model != ''
GROUP BY c.platform, m.model
ORDER BY c.platform, message_count DESC;
