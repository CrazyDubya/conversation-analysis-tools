-- Drop the existing model_usage view
DROP VIEW IF EXISTS model_usage;

-- Create the corrected model_usage view with proper join to conversations table
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
