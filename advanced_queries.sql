-- 1. Conversation Activity Over Time
SELECT 
    platform,
    strftime('%Y-%m', activity_date) AS month,
    SUM(conversations) AS total_conversations
FROM time_activity
GROUP BY platform, month
ORDER BY month, platform;

-- 2. Average Response Length Comparison
SELECT 
    c.platform,
    AVG(LENGTH(m.content)) AS avg_human_message_length,
    AVG(LENGTH(a.content)) AS avg_assistant_message_length,
    AVG(LENGTH(a.content)) / AVG(LENGTH(m.content)) AS response_ratio
FROM messages m
JOIN messages a ON m.conversation_id = a.conversation_id
JOIN conversations c ON m.conversation_id = c.id
WHERE m.sender = 'human' AND a.sender = 'assistant'
GROUP BY c.platform;

-- 3. Most Active Conversations
SELECT 
    conversation_id,
    title,
    platform,
    message_count,
    human_messages,
    assistant_messages,
    conversation_duration_minutes,
    ROUND(conversation_duration_minutes / 60.0, 2) AS conversation_hours
FROM conversation_summary
ORDER BY message_count DESC
LIMIT 20;

-- 4. Content Pattern Analysis - AI Topics
SELECT 
    c.id,
    c.title,
    c.platform,
    COUNT(m.id) AS matching_messages
FROM conversations c
JOIN messages m ON c.id = m.conversation_id
WHERE m.content LIKE '%machine learning%' OR m.content LIKE '%AI%' OR m.content LIKE '%neural network%'
GROUP BY c.id
HAVING matching_messages > 1
ORDER BY matching_messages DESC;

-- 5. Content Pattern Analysis - Code Discussions
SELECT 
    c.id,
    c.title,
    c.platform,
    COUNT(m.id) AS matching_messages
FROM conversations c
JOIN messages m ON c.id = m.conversation_id
WHERE m.content LIKE '%```python%' OR m.content LIKE '%```javascript%' OR m.content LIKE '%```java%' 
      OR m.content LIKE '%```c++%' OR m.content LIKE '%```sql%'
GROUP BY c.id
HAVING matching_messages > 1
ORDER BY matching_messages DESC;

-- 6. Content Pattern Analysis - Educational Topics
SELECT 
    c.id,
    c.title,
    c.platform,
    COUNT(m.id) AS matching_messages
FROM conversations c
JOIN messages m ON c.id = m.conversation_id
WHERE m.content LIKE '%teach%' OR m.content LIKE '%learn%' OR m.content LIKE '%explain%' 
      OR m.content LIKE '%tutorial%' OR m.content LIKE '%course%'
GROUP BY c.id
HAVING matching_messages > 1
ORDER BY matching_messages DESC;

-- 7. Short vs Long Conversations Analysis
SELECT
    platform,
    COUNT(CASE WHEN message_count <= 2 THEN 1 END) AS short_convos,
    COUNT(CASE WHEN message_count > 2 AND message_count <= 10 THEN 1 END) AS medium_convos,
    COUNT(CASE WHEN message_count > 10 THEN 1 END) AS long_convos,
    COUNT(*) AS total_convos,
    ROUND(AVG(message_count), 2) AS avg_messages_per_convo
FROM conversation_summary
GROUP BY platform;

-- 8. Analysis of Conversation Durations
SELECT
    platform,
    COUNT(CASE WHEN conversation_duration_minutes <= 1 THEN 1 END) AS instant_convos,
    COUNT(CASE WHEN conversation_duration_minutes > 1 AND conversation_duration_minutes <= 10 THEN 1 END) AS short_convos,
    COUNT(CASE WHEN conversation_duration_minutes > 10 AND conversation_duration_minutes <= 60 THEN 1 END) AS medium_convos,
    COUNT(CASE WHEN conversation_duration_minutes > 60 THEN 1 END) AS long_convos,
    ROUND(AVG(conversation_duration_minutes), 2) AS avg_duration_minutes
FROM conversation_summary
WHERE conversation_duration_minutes > 0
GROUP BY platform;

-- 9. Response Time Analysis
SELECT
    mp.conversation_id,
    c.platform,
    AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 * 60) AS avg_response_time_seconds
FROM message_pairs mp
JOIN conversations c ON mp.conversation_id = c.id
GROUP BY mp.conversation_id, c.platform
ORDER BY avg_response_time_seconds DESC;

-- 10. Platform Response Time Comparison
SELECT
    c.platform,
    ROUND(AVG((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 * 60), 2) AS avg_response_time_seconds,
    ROUND(MIN((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 * 60), 2) AS min_response_time_seconds,
    ROUND(MAX((julianday(mp.assistant_timestamp) - julianday(mp.human_timestamp)) * 24 * 60 * 60), 2) AS max_response_time_seconds
FROM message_pairs mp
JOIN conversations c ON mp.conversation_id = c.id
GROUP BY c.platform;
