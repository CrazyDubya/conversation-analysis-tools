//
//  GPT_Analyze2Tests.swift
//  GPT_Analyze2Tests
//
//  Created by Stephen Thompson on 5/29/24.
//

import XCTest
@testable import GPT_Analyze2

final class GPT_Analyze2Tests: XCTestCase {

    // MARK: - AnalysisError Tests

    func testAnalysisErrorDescriptions() {
        // Test file too large error
        let fileTooLargeError = AnalysisError.fileTooLarge(size: 150_000_000, maxSize: 100_000_000)
        XCTAssertNotNil(fileTooLargeError.errorDescription)
        XCTAssertTrue(fileTooLargeError.errorDescription!.contains("150.0"))
        XCTAssertTrue(fileTooLargeError.errorDescription!.contains("100"))

        // Test invalid JSON format error
        let invalidJSONError = AnalysisError.invalidJSONFormat
        XCTAssertNotNil(invalidJSONError.errorDescription)
        XCTAssertTrue(invalidJSONError.errorDescription!.contains("Invalid JSON"))

        // Test no messages found error
        let noMessagesError = AnalysisError.noMessagesFound
        XCTAssertNotNil(noMessagesError.errorDescription)
        XCTAssertTrue(noMessagesError.errorDescription!.contains("No messages"))

        // Test cancelled error
        let cancelledError = AnalysisError.cancelled
        XCTAssertNotNil(cancelledError.errorDescription)
        XCTAssertTrue(cancelledError.errorDescription!.contains("cancelled"))

        // Test write error
        let writeError = AnalysisError.writeError(path: "/test/path", underlying: NSError(domain: "test", code: 1))
        XCTAssertNotNil(writeError.errorDescription)
        XCTAssertTrue(writeError.errorDescription!.contains("/test/path"))
    }

    // MARK: - AnalysisResult Tests

    func testAnalysisResultSentimentDescriptions() {
        // Test positive sentiment
        let positiveResult = createMockResult(sentimentScore: 0.5)
        XCTAssertEqual(positiveResult.sentimentDescription, "Positive")

        // Test neutral sentiment (upper bound)
        let neutralHighResult = createMockResult(sentimentScore: 0.29)
        XCTAssertEqual(neutralHighResult.sentimentDescription, "Neutral")

        // Test neutral sentiment (lower bound)
        let neutralLowResult = createMockResult(sentimentScore: -0.29)
        XCTAssertEqual(neutralLowResult.sentimentDescription, "Neutral")

        // Test negative sentiment
        let negativeResult = createMockResult(sentimentScore: -0.5)
        XCTAssertEqual(negativeResult.sentimentDescription, "Negative")

        // Test boundary: exactly 0.3 is positive
        let boundaryPositiveResult = createMockResult(sentimentScore: 0.3)
        XCTAssertEqual(boundaryPositiveResult.sentimentDescription, "Positive")

        // Test boundary: exactly -0.3 is neutral
        let boundaryNeutralResult = createMockResult(sentimentScore: -0.3)
        XCTAssertEqual(boundaryNeutralResult.sentimentDescription, "Neutral")
    }

    func testAnalysisResultIdentifiable() {
        let result1 = createMockResult(sentimentScore: 0.0)
        let result2 = createMockResult(sentimentScore: 0.0)

        // Each result should have a unique ID
        XCTAssertNotEqual(result1.id, result2.id)
    }

    // MARK: - TextDocument Tests

    func testTextDocumentInitialization() {
        let document = TextDocument(text: "Test content")
        XCTAssertEqual(document.text, "Test content")
    }

    func testTextDocumentEmptyInitialization() {
        let document = TextDocument(text: "")
        XCTAssertEqual(document.text, "")
    }

    func testTextDocumentReadableContentTypes() {
        XCTAssertFalse(TextDocument.readableContentTypes.isEmpty)
    }

    // MARK: - Analyzer Tests

    @MainActor
    func testAnalyzerInitialState() {
        let analyzer = Analyzer()

        XCTAssertFalse(analyzer.isAnalyzing)
        XCTAssertEqual(analyzer.progress, 0.0)
        XCTAssertNil(analyzer.result)
        XCTAssertNil(analyzer.error)
        XCTAssertFalse(analyzer.statusText.isEmpty)
    }

    @MainActor
    func testAnalyzerCancelResetsState() {
        let analyzer = Analyzer()

        // Simulate some state
        analyzer.cancel()

        XCTAssertFalse(analyzer.isAnalyzing)
        XCTAssertEqual(analyzer.progress, 0.0)
        XCTAssertEqual(analyzer.statusText, "Analysis cancelled")
    }

    @MainActor
    func testAnalyzerWithInvalidFile() async {
        let analyzer = Analyzer()
        let invalidURL = URL(fileURLWithPath: "/nonexistent/file.json")

        analyzer.analyze(fileURL: invalidURL)

        // Wait for async operation to complete
        try? await Task.sleep(nanoseconds: 500_000_000) // 0.5 seconds

        XCTAssertFalse(analyzer.isAnalyzing)
        XCTAssertNotNil(analyzer.error)
    }

    @MainActor
    func testAnalyzerWithMalformedJSON() async throws {
        // Create a temporary file with invalid JSON
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_malformed.json")

        let malformedJSON = "{ invalid json content"
        try malformedJSON.write(to: tempFile, atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: tempFile)
        }

        let analyzer = Analyzer()
        analyzer.analyze(fileURL: tempFile)

        // Wait for async operation to complete
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second

        XCTAssertFalse(analyzer.isAnalyzing)
        XCTAssertNotNil(analyzer.error)
    }

    @MainActor
    func testAnalyzerWithEmptyConversations() async throws {
        // Create a temporary file with valid JSON but no messages
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_empty.json")

        let emptyConversations = "[]"
        try emptyConversations.write(to: tempFile, atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: tempFile)
        }

        let analyzer = Analyzer()
        analyzer.analyze(fileURL: tempFile)

        // Wait for async operation to complete
        try await Task.sleep(nanoseconds: 1_000_000_000) // 1 second

        XCTAssertFalse(analyzer.isAnalyzing)
        // Should error because no messages found
        XCTAssertNotNil(analyzer.error)
    }

    @MainActor
    func testAnalyzerWithValidChatGPTExport() async throws {
        // Create a mock ChatGPT export
        let tempDir = FileManager.default.temporaryDirectory
        let tempFile = tempDir.appendingPathComponent("test_valid.json")

        let validExport = """
        [
            {
                "title": "Test Conversation",
                "mapping": {
                    "node1": {
                        "message": {
                            "content": {
                                "parts": ["Hello world", "This is a test message"]
                            }
                        }
                    },
                    "node2": {
                        "message": {
                            "content": {
                                "parts": ["Testing word frequency analysis"]
                            }
                        }
                    }
                }
            }
        ]
        """
        try validExport.write(to: tempFile, atomically: true, encoding: .utf8)

        defer {
            try? FileManager.default.removeItem(at: tempFile)
        }

        let analyzer = Analyzer()
        analyzer.analyze(fileURL: tempFile)

        // Wait for async operation to complete
        try await Task.sleep(nanoseconds: 2_000_000_000) // 2 seconds

        XCTAssertFalse(analyzer.isAnalyzing)
        XCTAssertNil(analyzer.error)
        XCTAssertNotNil(analyzer.result)

        if let result = analyzer.result {
            XCTAssertGreaterThan(result.totalWords, 0)
            XCTAssertGreaterThan(result.uniqueWords, 0)
            XCTAssertFalse(result.topWords.isEmpty)
            XCTAssertGreaterThan(result.analysisTime, 0)
        }
    }

    // MARK: - Performance Tests

    func testWordCountingPerformance() throws {
        // Generate a large array of words for performance testing
        let words = (0..<10000).map { _ in ["hello", "world", "test", "swift", "analyzer"].randomElement()! }

        measure {
            let wordCounts = NSCountedSet(array: words)
            _ = wordCounts.allObjects
                .compactMap { $0 as? String }
                .sorted { wordCounts.count(for: $0) > wordCounts.count(for: $1) }
        }
    }

    // MARK: - Helper Methods

    private func createMockResult(sentimentScore: Double) -> AnalysisResult {
        AnalysisResult(
            fileName: "test.json",
            timestamp: Date(),
            totalWords: 100,
            uniqueWords: 50,
            topWords: [("test", 10, 10.0)],
            topWordsFiltered: [("test", 10, 10.0)],
            sentimentScore: sentimentScore,
            analysisTime: 1.0
        )
    }
}
