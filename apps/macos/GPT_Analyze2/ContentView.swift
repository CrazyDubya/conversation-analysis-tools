import SwiftUI
import AppKit
import Foundation
import NaturalLanguage
import UniformTypeIdentifiers

// MARK: - Constants

private enum AnalysisConstants {
    static let maxFileSizeBytes: Int = 100_000_000 // 100MB
    static let maxWordsToDisplay: Int = 100
    static let maxWordsToExport: Int = 100_000
    static let defaultOutputFileName = "gpt_analysis_results"
}

// MARK: - Error Types

enum AnalysisError: LocalizedError {
    case fileTooLarge(size: Int, maxSize: Int)
    case invalidJSONFormat
    case noMessagesFound
    case cancelled
    case writeError(path: String, underlying: Error)

    var errorDescription: String? {
        switch self {
        case .fileTooLarge(let size, let maxSize):
            let sizeMB = Double(size) / 1_000_000
            let maxMB = Double(maxSize) / 1_000_000
            return "File too large (\(String(format: "%.1f", sizeMB))MB). Maximum size is \(String(format: "%.0f", maxMB))MB."
        case .invalidJSONFormat:
            return "Invalid JSON format. Please select a valid ChatGPT export file."
        case .noMessagesFound:
            return "No messages found in the file. The file may be empty or have an unexpected structure."
        case .cancelled:
            return "Analysis was cancelled."
        case .writeError(let path, let underlying):
            return "Failed to write results to \(path): \(underlying.localizedDescription)"
        }
    }
}

// MARK: - Analysis Result Model

struct AnalysisResult: Identifiable {
    let id = UUID()
    let fileName: String
    let timestamp: Date
    let totalWords: Int
    let uniqueWords: Int
    let topWords: [(word: String, count: Int, percentage: Double)]
    let topWordsFiltered: [(word: String, count: Int, percentage: Double)]
    let sentimentScore: Double
    let analysisTime: TimeInterval

    var sentimentDescription: String {
        switch sentimentScore {
        case 0.3...: return "Positive"
        case -0.3..<0.3: return "Neutral"
        default: return "Negative"
        }
    }

    var sentimentColor: Color {
        switch sentimentScore {
        case 0.3...: return .green
        case -0.3..<0.3: return .secondary
        default: return .red
        }
    }
}

// MARK: - Analyzer

@MainActor
class Analyzer: ObservableObject {
    @Published var statusText: String = "Select a ChatGPT export file to analyze"
    @Published var isAnalyzing: Bool = false
    @Published var progress: Double = 0.0
    @Published var result: AnalysisResult?
    @Published var error: AnalysisError?

    private var analysisTask: Task<Void, Never>?

    private static let stopWords: Set<String> = [
        "a", "an", "the", "and", "or", "but", "because", "as", "if", "when",
        "while", "of", "at", "by", "for", "with", "about", "against", "between",
        "into", "through", "during", "before", "after", "above", "below", "to",
        "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "all", "any", "both", "each",
        "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
        "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
        "don", "should", "now", "i", "you", "he", "she", "it", "we", "they", "me",
        "him", "her", "us", "them", "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "would", "could", "might",
        "must", "shall", "may", "need", "dare", "ought", "used", "what", "which",
        "who", "whom", "whose", "where", "how", "why", "am", "isn", "aren", "wasn",
        "weren", "hasn", "haven", "hadn", "doesn", "didn", "won", "wouldn", "couldn",
        "shouldn", "mightn", "mustn", "let", "that's", "there's", "here's", "it's",
        "i'm", "you're", "we're", "they're", "i've", "you've", "we've", "they've",
        "i'd", "you'd", "he'd", "she'd", "we'd", "they'd", "i'll", "you'll", "he'll",
        "she'll", "we'll", "they'll", "isn't", "aren't", "wasn't", "weren't", "hasn't",
        "haven't", "hadn't", "doesn't", "didn't", "won't", "wouldn't", "couldn't",
        "shouldn't", "mightn't", "mustn't", "let's", "also", "like", "get", "got",
        "really", "know", "think", "make", "want", "see", "way", "well", "back",
        "even", "new", "one", "two", "first", "last", "long", "great", "little",
        "own", "other", "old", "right", "big", "high", "different", "small", "large",
        "next", "early", "young", "important", "few", "public", "bad", "same", "able"
    ]

    func cancel() {
        analysisTask?.cancel()
        analysisTask = nil
        isAnalyzing = false
        progress = 0.0
        statusText = "Analysis cancelled"
    }

    func analyze(fileURL: URL) {
        analysisTask?.cancel()

        isAnalyzing = true
        progress = 0.0
        error = nil
        result = nil

        analysisTask = Task {
            await performAnalysis(fileURL: fileURL)
        }
    }

    private func performAnalysis(fileURL: URL) async {
        let startTime = Date()
        let fileName = fileURL.lastPathComponent

        do {
            // Step 1: Validate file size
            statusText = "Checking file size..."
            progress = 0.05

            let attributes = try FileManager.default.attributesOfItem(atPath: fileURL.path)
            guard let fileSize = attributes[.size] as? Int else {
                throw AnalysisError.invalidJSONFormat
            }

            if fileSize > AnalysisConstants.maxFileSizeBytes {
                throw AnalysisError.fileTooLarge(size: fileSize, maxSize: AnalysisConstants.maxFileSizeBytes)
            }

            try Task.checkCancellation()

            // Step 2: Load and parse JSON
            statusText = "Loading file..."
            progress = 0.1

            let data = try Data(contentsOf: fileURL)

            try Task.checkCancellation()

            statusText = "Parsing JSON..."
            progress = 0.2

            guard let conversationsData = try JSONSerialization.jsonObject(with: data, options: []) as? [[String: Any]] else {
                throw AnalysisError.invalidJSONFormat
            }

            try Task.checkCancellation()

            // Step 3: Extract messages
            statusText = "Extracting messages..."
            progress = 0.3

            var messages: [String] = []
            for conversation in conversationsData {
                if let mapping = conversation["mapping"] as? [String: [String: Any]] {
                    for node in mapping.values {
                        if let message = node["message"] as? [String: Any],
                           let content = message["content"] as? [String: Any],
                           let parts = content["parts"] as? [String] {
                            messages.append(contentsOf: parts)
                        }
                    }
                }
            }

            guard !messages.isEmpty else {
                throw AnalysisError.noMessagesFound
            }

            try Task.checkCancellation()

            // Step 4: Tokenize text
            statusText = "Tokenizing text..."
            progress = 0.4

            let allText = messages.joined(separator: " ")
            let words = await tokenize(text: allText)

            try Task.checkCancellation()

            // Step 5: Count word frequencies
            statusText = "Counting word frequencies..."
            progress = 0.6

            let wordCounts = NSCountedSet(array: words)
            let totalWordCount = words.count
            let uniqueWordCount = wordCounts.count

            let sortedWords = wordCounts.allObjects
                .compactMap { $0 as? String }
                .sorted { wordCounts.count(for: $0) > wordCounts.count(for: $1) }

            try Task.checkCancellation()

            // Step 6: Filter stop words and count again
            statusText = "Filtering stop words..."
            progress = 0.7

            let filteredWords = words.filter { !Self.stopWords.contains($0) }
            let filteredWordCounts = NSCountedSet(array: filteredWords)
            let filteredTotalCount = filteredWords.count

            let filteredSortedWords = filteredWordCounts.allObjects
                .compactMap { $0 as? String }
                .sorted { filteredWordCounts.count(for: $0) > filteredWordCounts.count(for: $1) }

            try Task.checkCancellation()

            // Step 7: Analyze sentiment
            statusText = "Analyzing sentiment..."
            progress = 0.85

            let sentimentScore = await analyzeSentiment(text: allText)

            try Task.checkCancellation()

            // Step 8: Build results
            statusText = "Building results..."
            progress = 0.95

            let topWords: [(String, Int, Double)] = sortedWords.prefix(AnalysisConstants.maxWordsToDisplay).map { word in
                let count = wordCounts.count(for: word)
                let percentage = totalWordCount > 0 ? (Double(count) / Double(totalWordCount)) * 100 : 0
                return (word, count, percentage)
            }

            let topWordsFiltered: [(String, Int, Double)] = filteredSortedWords.prefix(AnalysisConstants.maxWordsToDisplay).map { word in
                let count = filteredWordCounts.count(for: word)
                let percentage = filteredTotalCount > 0 ? (Double(count) / Double(filteredTotalCount)) * 100 : 0
                return (word, count, percentage)
            }

            let endTime = Date()
            let analysisTime = endTime.timeIntervalSince(startTime)

            let analysisResult = AnalysisResult(
                fileName: fileName,
                timestamp: endTime,
                totalWords: totalWordCount,
                uniqueWords: uniqueWordCount,
                topWords: topWords,
                topWordsFiltered: topWordsFiltered,
                sentimentScore: sentimentScore,
                analysisTime: analysisTime
            )

            result = analysisResult
            progress = 1.0
            statusText = "Analysis complete in \(String(format: "%.2f", analysisTime)) seconds"
            isAnalyzing = false

        } catch is CancellationError {
            error = .cancelled
            statusText = "Analysis cancelled"
            isAnalyzing = false
            progress = 0.0
        } catch let analysisError as AnalysisError {
            error = analysisError
            statusText = analysisError.localizedDescription
            isAnalyzing = false
            progress = 0.0
        } catch {
            self.error = .invalidJSONFormat
            statusText = "Error: \(error.localizedDescription)"
            isAnalyzing = false
            progress = 0.0
        }
    }

    private func tokenize(text: String) async -> [String] {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let tokenizer = NLTokenizer(unit: .word)
                tokenizer.string = text
                var words: [String] = []
                words.reserveCapacity(text.count / 5) // Estimate average word length

                tokenizer.enumerateTokens(in: text.startIndex..<text.endIndex) { tokenRange, _ in
                    let word = String(text[tokenRange]).lowercased()
                    if word.count > 1 || word == "i" || word == "a" { // Filter single chars except common words
                        words.append(word)
                    }
                    return true
                }

                continuation.resume(returning: words)
            }
        }
    }

    private func analyzeSentiment(text: String) async -> Double {
        await withCheckedContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let tagger = NLTagger(tagSchemes: [.sentimentScore])
                tagger.string = text

                var totalScore: Double = 0
                var count: Int = 0

                // Analyze sentiment across multiple segments for better accuracy
                tagger.enumerateTags(in: text.startIndex..<text.endIndex, unit: .paragraph, scheme: .sentimentScore) { tag, _ in
                    if let tag = tag, let score = Double(tag.rawValue) {
                        totalScore += score
                        count += 1
                    }
                    return true
                }

                let averageScore = count > 0 ? totalScore / Double(count) : 0.0
                continuation.resume(returning: averageScore)
            }
        }
    }

    func exportResults(to url: URL) async throws {
        guard let result = result else { return }

        var content = """
        GPT-Analyze2 Results
        ====================
        File: \(result.fileName)
        Analyzed: \(result.timestamp.formatted())
        Analysis Time: \(String(format: "%.2f", result.analysisTime)) seconds

        Summary
        -------
        Total Words: \(result.totalWords.formatted())
        Unique Words: \(result.uniqueWords.formatted())
        Sentiment: \(result.sentimentDescription) (\(String(format: "%.3f", result.sentimentScore)))

        Top \(result.topWords.count) Words (All)
        -----------------------

        """

        for (index, item) in result.topWords.enumerated() {
            content += "\(index + 1). \(item.word): \(item.count) (\(String(format: "%.2f", item.percentage))%)\n"
        }

        content += """

        Top \(result.topWordsFiltered.count) Words (Excluding Stop Words)
        -----------------------------------------

        """

        for (index, item) in result.topWordsFiltered.enumerated() {
            content += "\(index + 1). \(item.word): \(item.count) (\(String(format: "%.2f", item.percentage))%)\n"
        }

        do {
            try content.write(to: url, atomically: true, encoding: .utf8)
        } catch {
            throw AnalysisError.writeError(path: url.path, underlying: error)
        }
    }
}

// MARK: - Content View

struct ContentView: View {
    @StateObject private var analyzer = Analyzer()
    @State private var showingExportPanel = false

    var body: some View {
        NavigationSplitView {
            sidebarView
        } detail: {
            detailView
        }
        .frame(minWidth: 800, minHeight: 600)
    }

    // MARK: - Sidebar

    private var sidebarView: some View {
        VStack(spacing: 20) {
            headerSection

            Divider()

            actionSection

            if analyzer.isAnalyzing {
                progressSection
            }

            Spacer()

            if let result = analyzer.result {
                summarySection(result: result)
            }
        }
        .padding()
        .frame(minWidth: 280)
    }

    private var headerSection: some View {
        VStack(spacing: 8) {
            Image(systemName: "doc.text.magnifyingglass")
                .font(.system(size: 48))
                .foregroundColor(.accentColor)

            Text("GPT Analyzer")
                .font(.title)
                .fontWeight(.bold)

            Text("Analyze ChatGPT Conversations")
                .font(.subheadline)
                .foregroundColor(.secondary)
        }
        .padding(.top)
    }

    private var actionSection: some View {
        VStack(spacing: 12) {
            Button(action: selectFile) {
                Label("Select Export File", systemImage: "folder.badge.plus")
                    .frame(maxWidth: .infinity)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(analyzer.isAnalyzing)

            if analyzer.isAnalyzing {
                Button(action: analyzer.cancel) {
                    Label("Cancel", systemImage: "xmark.circle")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
                .tint(.red)
            }

            if analyzer.result != nil {
                Button(action: { showingExportPanel = true }) {
                    Label("Export Results", systemImage: "square.and.arrow.up")
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .controlSize(.regular)
            }
        }
        .fileExporter(
            isPresented: $showingExportPanel,
            document: TextDocument(text: ""),
            contentType: .plainText,
            defaultFilename: "\(AnalysisConstants.defaultOutputFileName).txt"
        ) { result in
            if case .success(let url) = result {
                Task {
                    try? await analyzer.exportResults(to: url)
                }
            }
        }
    }

    private var progressSection: some View {
        VStack(spacing: 8) {
            ProgressView(value: analyzer.progress)
                .progressViewStyle(.linear)

            Text(analyzer.statusText)
                .font(.caption)
                .foregroundColor(.secondary)
                .lineLimit(2)
                .multilineTextAlignment(.center)
        }
        .padding(.vertical)
    }

    private func summarySection(result: AnalysisResult) -> some View {
        VStack(spacing: 12) {
            Divider()

            Text("Summary")
                .font(.headline)

            Grid(alignment: .leading, horizontalSpacing: 12, verticalSpacing: 8) {
                GridRow {
                    Text("Total Words")
                        .foregroundColor(.secondary)
                    Text(result.totalWords.formatted())
                        .fontWeight(.medium)
                }
                GridRow {
                    Text("Unique Words")
                        .foregroundColor(.secondary)
                    Text(result.uniqueWords.formatted())
                        .fontWeight(.medium)
                }
                GridRow {
                    Text("Sentiment")
                        .foregroundColor(.secondary)
                    HStack(spacing: 4) {
                        Text(result.sentimentDescription)
                            .fontWeight(.medium)
                            .foregroundColor(result.sentimentColor)
                        Text("(\(String(format: "%.2f", result.sentimentScore)))")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }
                }
                GridRow {
                    Text("Analysis Time")
                        .foregroundColor(.secondary)
                    Text("\(String(format: "%.2f", result.analysisTime))s")
                        .fontWeight(.medium)
                }
            }
            .font(.callout)
        }
        .padding(.bottom)
    }

    // MARK: - Detail View

    private var detailView: some View {
        Group {
            if let error = analyzer.error {
                errorView(error: error)
            } else if let result = analyzer.result {
                resultsView(result: result)
            } else {
                emptyStateView
            }
        }
    }

    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "arrow.left.circle")
                .font(.system(size: 64))
                .foregroundColor(.secondary.opacity(0.5))

            Text("Select a File to Begin")
                .font(.title2)
                .foregroundColor(.secondary)

            Text("Choose a ChatGPT JSON export file to analyze word frequencies and sentiment.")
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func errorView(error: AnalysisError) -> some View {
        VStack(spacing: 16) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 64))
                .foregroundColor(.orange)

            Text("Analysis Error")
                .font(.title2)
                .fontWeight(.semibold)

            Text(error.localizedDescription)
                .font(.body)
                .foregroundColor(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 400)

            Button("Try Again") {
                analyzer.error = nil
            }
            .buttonStyle(.bordered)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private func resultsView(result: AnalysisResult) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                // Header
                VStack(alignment: .leading, spacing: 4) {
                    Text("Analysis Results")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text(result.fileName)
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }

                // Word frequency sections
                HStack(alignment: .top, spacing: 24) {
                    wordListSection(
                        title: "Top Words (All)",
                        words: result.topWords,
                        icon: "textformat.abc"
                    )

                    wordListSection(
                        title: "Top Words (Filtered)",
                        words: result.topWordsFiltered,
                        icon: "line.3.horizontal.decrease.circle"
                    )
                }
            }
            .padding(24)
        }
    }

    private func wordListSection(title: String, words: [(word: String, count: Int, percentage: Double)], icon: String) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Label(title, systemImage: icon)
                .font(.headline)

            VStack(spacing: 0) {
                ForEach(Array(words.enumerated()), id: \.offset) { index, item in
                    HStack {
                        Text("\(index + 1).")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .frame(width: 30, alignment: .trailing)

                        Text(item.word)
                            .fontWeight(.medium)

                        Spacer()

                        Text("\(item.count)")
                            .foregroundColor(.secondary)

                        Text("(\(String(format: "%.1f", item.percentage))%)")
                            .font(.caption)
                            .foregroundColor(.secondary)
                            .frame(width: 60, alignment: .trailing)
                    }
                    .padding(.vertical, 6)
                    .padding(.horizontal, 8)
                    .background(index % 2 == 0 ? Color.secondary.opacity(0.05) : Color.clear)
                }
            }
            .background(Color.secondary.opacity(0.03))
            .cornerRadius(8)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    // MARK: - Actions

    private func selectFile() {
        let panel = NSOpenPanel()
        panel.allowedContentTypes = [.json]
        panel.allowsMultipleSelection = false
        panel.canChooseDirectories = false
        panel.message = "Select a ChatGPT export file (JSON format)"
        panel.prompt = "Analyze"

        if panel.runModal() == .OK, let url = panel.url {
            analyzer.analyze(fileURL: url)
        }
    }
}

// MARK: - Text Document for Export

struct TextDocument: FileDocument {
    static var readableContentTypes: [UTType] { [.plainText] }

    var text: String

    init(text: String) {
        self.text = text
    }

    init(configuration: ReadConfiguration) throws {
        if let data = configuration.file.regularFileContents {
            text = String(decoding: data, as: UTF8.self)
        } else {
            text = ""
        }
    }

    func fileWrapper(configuration: WriteConfiguration) throws -> FileWrapper {
        FileWrapper(regularFileWithContents: Data(text.utf8))
    }
}

// MARK: - Preview

#Preview {
    ContentView()
}
