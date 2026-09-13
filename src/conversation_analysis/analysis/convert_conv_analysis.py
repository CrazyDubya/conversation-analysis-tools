# Ported from chatgpt-archive-clean/chatgptarchive-og/convert_conv_analysis.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: byte-identical to ai-conversation-analyzer/convert_conv_analysis.py; kept the og copy

import re
import xml.etree.ElementTree as ET

import pandas as pd


def extract_xml_flexible(content_block_text):
    match = re.search(r"<(\?xml|[a-zA-Z]+)[^>]*>.*<\/\1>", content_block_text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return ""


def is_valid_xml(xml_text):
    try:
        ET.fromstring(xml_text)
        return True
    except ET.ParseError:
        return False


def parse_xml_content(xml_text):
    try:
        root = ET.fromstring(xml_text)
        parsed_data = {}

        if root.find('Keywords') is not None:
            parsed_data['Keywords'] = [kw.text for kw in root.findall('.//Keyword')]

        if root.find('Summary') is not None:
            parsed_data['Summary'] = root.find('Summary').text

        if root.find('SentimentTrends') is not None:
            trend = root.find('.//Trend')
            if trend is not None:
                parsed_data['SentimentTrends'] = {
                    'ConversationId': trend.find('ConversationId').text if trend.find(
                        'ConversationId') is not None else None,
                    'Timestamp': trend.find('Timestamp').text if trend.find('Timestamp') is not None else None,
                    'Sentiment': trend.find('Sentiment').text if trend.find('Sentiment') is not None else None
                }

        if root.find('WordFrequencies') is not None:
            parsed_data['WordFrequencies'] = [
                {'Text': word.find('Text').text if word.find('Text') is not None else None,
                 'Frequency': word.find('Frequency').text if word.find('Frequency') is not None else None}
                for word in root.findall('.//Word')
            ]

        if root.find('Anomalies') is not None:
            anomaly = root.find('.//Anomaly')
            if anomaly is not None:
                parsed_data['Anomalies'] = {
                    'ConversationId': anomaly.find('ConversationId').text if anomaly.find(
                        'ConversationId') is not None else None,
                    'MessageCount': anomaly.find('MessageCount').text if anomaly.find(
                        'MessageCount') is not None else None,
                    'IsAnomaly': anomaly.find('IsAnomaly').text if anomaly.find('IsAnomaly') is not None else None
                }

        if root.find('Statistics') is not None:
            stats = root.find('Statistics').find('MessageLengths')
            if stats is not None:
                parsed_data['Statistics'] = {
                    'Longest': stats.find('Longest').text if stats.find('Longest') is not None else None,
                    'Shortest': stats.find('Shortest').text if stats.find('Shortest') is not None else None,
                    'Mean': stats.find('Mean').text if stats.find('Mean') is not None else None,
                    'Median': stats.find('Median').text if stats.find('Median') is not None else None,
                    'StdDev': stats.find('StdDev').text if stats.find('StdDev') is not None else None
                }

        return parsed_data

    except ET.ParseError:
        return {}


# Load the CSV file
file_path = 'conversation_analysis.csv'
data = pd.read_csv(file_path)

# Extract and parse the XML content for each row
parsed_data_list = []
error_rows = []

for index, row in data.iterrows():
    xml_text = extract_xml_flexible(row['xml'])
    if xml_text and is_valid_xml(xml_text):
        try:
            parsed_data = parse_xml_content(xml_text)
            parsed_data['conversation_id'] = row['conversation_id']
            parsed_data_list.append(parsed_data)
        except Exception as e:
            error_rows.append((index, str(e)))
    else:
        error_rows.append((index, "Invalid XML or Extraction Failed"))

# Convert the parsed data into a DataFrame
final_structured_data = pd.DataFrame(parsed_data_list)

# Convert lists of keywords to a comma-separated string for readability
final_structured_data['Keywords'] = final_structured_data['Keywords'].apply(
    lambda x: ', '.join(x) if isinstance(x, list) else x)

# Save the new formatted data to a CSV file
output_file_path = 'formatted_conversation_analysis.csv'
final_structured_data.to_csv(output_file_path, index=False)

# Report any rows that caused errors
error_report_path = 'error_report.csv'
error_report = pd.DataFrame(error_rows, columns=['Row_Index', 'Error'])
error_report.to_csv(error_report_path, index=False)

print(f"Formatted data saved to {output_file_path}")
print(f"Error report saved to {error_report_path}")
