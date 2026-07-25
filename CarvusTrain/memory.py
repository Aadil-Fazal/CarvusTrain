"""Memory management system for knowledge retention, semantic search, code comprehension, KV caching, and context window tracking with learning validation.

Features:
- Pure-Python TF-IDF vectorizer for semantic search (no external dependencies required)
- Optional scikit-learn TfidfVectorizer integration for faster search
- Optional sentence-transformers for deep semantic embeddings
- Cosine similarity scoring with automatic index rebuilding
- Graceful fallback to word overlap when no index is built
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union


# ---------------------------------------------------------------------------
# Pure-Python TF-IDF vectorizer — zero external dependencies
# ---------------------------------------------------------------------------

class TfidfVectorizer:
    """Pure-Python TF-IDF vectorizer with vocabulary building, IDF computation,
    and sparse-dense vector transformation.

    Mimics sklearn.feature_extraction.text.TfidfVectorizer but works
    without any external packages.
    """

    def __init__(self, max_features: int = 5000) -> None:
        self.max_features = max_features
        self.vocabulary: Dict[str, int] = {}          # term -> index
        self.idf: List[float] = []                     # idf per term index
        self.doc_count: int = 0
        self._fitted: bool = False

    def _tokenize(self, text: str) -> List[str]:
        """Lowercase, split on non-alphanumeric, filter short tokens."""
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        return [t for t in tokens if len(t) >= 2]

    def _term_freq(self, tokens: List[str]) -> Dict[str, float]:
        """Return relative term frequency (count / doc_length)."""
        n = len(tokens)
        if n == 0:
            return {}
        counts: Dict[str, float] = {}
        for t in tokens:
            counts[t] = counts.get(t, 0.0) + 1.0
        return {t: c / n for t, c in counts.items()}

    def fit(self, documents: List[str]) -> "TfidfVectorizer":
        """Build vocabulary and compute IDF from a corpus of documents."""
        self.doc_count = len(documents)
        doc_freq: Dict[str, int] = {}

        for doc in documents:
            tokens = self._tokenize(doc)
            seen = set(tokens)
            for t in seen:
                doc_freq[t] = doc_freq.get(t, 0) + 1

        # Sort by document frequency descending, take top max_features
        sorted_terms = sorted(doc_freq.items(), key=lambda x: -x[1])[:self.max_features]
        self.vocabulary = {term: idx for idx, (term, _) in enumerate(sorted_terms)}
        self.idf = [0.0] * len(self.vocabulary)

        for term, idx in self.vocabulary.items():
            df = doc_freq.get(term, 1)
            self.idf[idx] = math.log((self.doc_count + 1) / (df + 1)) + 1.0

        self._fitted = True
        return self

    def transform(self, documents: List[str]) -> List[List[float]]:
        """Transform documents into TF-IDF vectors (list of lists, dense)."""
        if not self._fitted:
            return []
        vectors: List[List[float]] = []
        vocab_size = len(self.vocabulary)

        for doc in documents:
            vec = [0.0] * vocab_size
            tokens = self._tokenize(doc)
            tf = self._term_freq(tokens)
            for term, freq in tf.items():
                idx = self.vocabulary.get(term)
                if idx is not None:
                    vec[idx] = freq * self.idf[idx]
            vectors.append(vec)

        return vectors

    def fit_transform(self, documents: List[str]) -> List[List[float]]:
        """Fit and transform in one call."""
        self.fit(documents)
        return self.transform(documents)


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def _dot_product(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(v: List[float]) -> float:
    return math.sqrt(sum(x * x for x in v)) + 1e-12


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    return _dot_product(a, b) / (_norm(a) * _norm(b))


# ---------------------------------------------------------------------------
# English Grammar knowledge for natural-language understanding
# ---------------------------------------------------------------------------

ENGLISH_GRAMMAR: Dict[str, List[str]] = {
    "parts_of_speech": [
        "Nouns represent people, places, things, or ideas (e.g., dog, city, love, John).",
        "Pronouns replace nouns and include personal (I, you, he, she, it, we, they), possessive (my, your, his, her, its, our, their), reflexive (myself, yourself, himself), and relative (who, which, that) pronouns.",
        "Verbs describe actions, states, or occurrences. They conjugate by tense (walk, walked, walking), person (I walk, she walks), number, mood, and voice.",
        "Adjectives modify nouns or pronouns, describing qualities (beautiful, large, quick, interesting).",
        "Adverbs modify verbs, adjectives, or other adverbs, indicating manner (quickly), place (here), time (now), frequency (often), or degree (very).",
        "Prepositions show relationships between nouns/pronouns and other words (in, on, at, by, with, from, to, for, about, through, between, under, over).",
        "Conjunctions connect words, phrases, or clauses: coordinating (and, but, or, nor, for, yet, so), subordinating (because, although, while, since, unless, if, when), and correlative (either...or, neither...nor, not only...but also).",
        "Determiners and articles include definite (the), indefinite (a, an), demonstrative (this, that, these, those), possessive (my, your, his), quantifiers (some, many, few, several, each, every, all, no).",
        "Interjections express emotion (oh, wow, hey, ouch, ah, well).",
    ],
    "sentence_structure": [
        "A sentence typically follows Subject-Verb-Object (SVO) order in English: 'The cat (subject) chased (verb) the mouse (object)'.",
        "Simple sentences contain one independent clause: 'She runs every morning.'",
        "Compound sentences join two independent clauses with coordinating conjunctions: 'I wanted to go, but it was raining.'",
        "Complex sentences have one independent clause and at least one dependent clause: 'Although it was raining, we went for a walk.'",
        "Compound-complex sentences contain multiple independent clauses and at least one dependent clause.",
        "Declarative sentences make statements: 'The sun rises in the east.'",
        "Interrogative sentences ask questions: 'Where are you going?'",
        "Imperative sentences give commands: 'Please close the door.'",
        "Exclamatory sentences express strong emotion: 'What a beautiful day!'",
        "Appositives rename or describe nouns: 'My brother, a skilled programmer, built this app.'",
    ],
    "tenses": [
        "Present simple: 'I walk' — habits, facts, general truths.",
        "Present continuous: 'I am walking' — ongoing actions now or near future.",
        "Present perfect: 'I have walked' — past actions with present relevance.",
        "Present perfect continuous: 'I have been walking' — actions that started in past and continue.",
        "Past simple: 'I walked' — completed actions at a specific past time.",
        "Past continuous: 'I was walking' — ongoing actions in the past.",
        "Past perfect: 'I had walked' — actions completed before another past action.",
        "Past perfect continuous: 'I had been walking' — ongoing past actions before another past event.",
        "Future simple: 'I will walk' — predictions, promises, spontaneous decisions.",
        "Future continuous: 'I will be walking' — actions in progress at a future time.",
        "Future perfect: 'I will have walked' — actions completed by a future time.",
        "Future perfect continuous: 'I will have been walking' — ongoing actions up to a future point.",
    ],
    "grammar_rules": [
        "Subject-verb agreement: singular subjects take singular verbs, plural subjects take plural verbs. 'He runs' vs 'They run'.",
        "Articles: Use 'a' before consonant sounds, 'an' before vowel sounds, 'the' for specific references: 'a dog', 'an apple', 'the book on the table'.",
        "Pronoun-antecedent agreement: pronouns must agree in number and gender with their antecedents: 'Every student must bring his or her book'.",
        "Parallel structure: items in a list or comparison should have the same grammatical form: 'I like swimming, biking, and running' (not 'swimming, biking, and to run').",
        "Modifier placement: modifiers should be placed near the words they modify to avoid dangling or misplaced modifiers. 'Walking to school, the rain started' is incorrect; 'Walking to school, I felt the rain' is correct.",
        "Comma splices: do not join two independent clauses with only a comma. Use a semicolon, period, or conjunction: 'It is raining; I will stay inside' or 'It is raining, so I will stay inside'.",
        "Run-on sentences: separate independent clauses properly. 'I love coding it is fun' should be 'I love coding. It is fun.' or 'I love coding because it is fun.'",
        "Active voice is preferred over passive voice for clarity: 'The cat chased the mouse' (active) vs 'The mouse was chased by the cat' (passive).",
        "Double negatives create positive meaning or confusion: 'I don't have nothing' actually means 'I have something'. Use 'I don't have anything' or 'I have nothing'.",
        "Prepositions at end of sentences: While traditional grammar forbids ending sentences with prepositions, modern English allows it: 'Who are you talking to?' is acceptable.",
        "Split infinitives: Placing an adverb between 'to' and the verb (e.g., 'to boldly go') is grammatically acceptable in modern English.",
        "Conditional sentences: Type 0 (general truth: 'If you heat ice, it melts'), Type 1 (real future: 'If it rains, I will stay'), Type 2 (unreal present: 'If I were rich, I would travel'), Type 3 (unreal past: 'If I had studied, I would have passed').",
        "Reported speech changes tense: 'He said, \"I am tired\"' becomes 'He said (that) he was tired'.",
        "Relative clauses: defining (restrictive, no commas: 'The book that I read was good') and non-defining (non-restrictive, with commas: 'The book, which I read yesterday, was good').",
    ],
    "common_mistakes": [
        "Their/There/They're: 'Their' is possessive (their house), 'There' indicates place (there is), 'They're' is contraction of 'they are'.",
        "Your/You're: 'Your' is possessive (your book), 'You're' is contraction of 'you are'.",
        "Its/It's: 'Its' is possessive (its color), 'It's' is contraction of 'it is' or 'it has'.",
        "To/Too/Two: 'To' indicates direction, 'Too' means also or excessively, 'Two' is the number 2.",
        "Affect/Effect: 'Affect' is usually a verb (to influence), 'Effect' is usually a noun (a result).",
        "Then/Than: 'Then' refers to time, 'Than' is used for comparisons.",
        "Who/Whom: 'Who' is for subjects, 'Whom' is for objects. 'Who called you?' vs 'To whom did you speak?'",
        "Less/Fewer: 'Less' is for uncountable nouns (less water), 'Fewer' is for countable nouns (fewer books).",
        "Between/Among: Use 'between' for two items, 'among' for three or more.",
        "Lay/Lie: 'Lay' requires an object (lay the book down), 'Lie' does not (lie down).",
    ],
    "writing_style": [
        "Use clear, concise language. Avoid unnecessary jargon and wordiness.",
        "Vary sentence length and structure for better readability and flow.",
        "Use transitional words and phrases to connect ideas: however, therefore, moreover, consequently, in addition, for example, nevertheless.",
        "Maintain consistent point of view: first person (I, we), second person (you), or third person (he, she, it, they).",
        "Use concrete examples and specific details instead of vague generalizations.",
        "Organize ideas logically with clear topic sentences and supporting details.",
        "Avoid cliches, redundancies, and overused expressions.",
        "Proofread for spelling, grammar, and punctuation errors.",
        "Use appropriate tone for the audience and context: formal, informal, technical, or conversational.",
        "Read your writing aloud to catch awkward phrasing and run-on sentences.",
    ],
}


# Programming language keyword signatures for detection
PROGRAMMING_LANGUAGES: Dict[str, Set[str]] = {
    "python": {
        "def ", "class ", "import ", "from ", "print(", "if __name__", "return ",
        "lambda", "yield", "async def", "await", "with ", "as ", "elif", "else:",
        "try:", "except:", "finally:", "raise ", "pass", "None", "True", "False",
        "self", "__init__", "len()", "range()", "enumerate", "zip()", "map()",
        "filter()", "list comprehension", "dict comprehension", "@staticmethod",
        "@classmethod", "@property", "def __str__", "def __repr__",
    },
    "javascript": {
        "function", "const ", "let ", "var ", "=>", "console.log", "document.",
        "window.", "addEventListener", "querySelector", "fetch(", "Promise",
        "async ", "await ", "export ", "import ", "require(", "module.exports",
        "this.", "new ", "typeof", "instanceof", "=== ", "!== ", "=> {",
        "=> (", "class ", "extends ", "constructor(", "super(", "null",
        "undefined", "Math.", "JSON.", "Array.", "Object.",
    },
    "typescript": {
        "interface ", "type ", "enum ", ": string", ": number", ": boolean",
        ": void", ": any", ": never", "<T>", "<T,", "as ", "readonly",
        "public ", "private ", "protected ", "?: ", "!:", "| ", "& ",
        "extends ", "implements ", "namespace ", "declare ", "abstract ",
    },
    "java": {
        "public class", "private ", "protected ", "static void", "public static",
        "String[] args", "System.out", "new ", "import java.", "package ",
        "extends ", "implements ", "@Override", "@Deprecated", "interface ",
        "abstract class", "final ", "void ", "int ", "boolean ", "String ",
        "List<", "Map<", "Set<", "ArrayList", "HashMap", "try {", "catch(",
        "throws ", "this.", "super.", "return ",
    },
    "cpp": {
        "#include", "std::", "int main", "cout <<", "cin >>", "class ",
        "public:", "private:", "protected:", "virtual ", "override",
        "template <", "typename ", "auto ", "constexpr", "nullptr",
        "std::vector", "std::map", "std::string", "std::cout", "std::cin",
        "->", "::", "new ", "delete ", "throw ", "catch ", "namespace ",
        "using namespace", "friend ", "struct ",
    },
    "c": {
        "#include", "int main", "printf(", "scanf(", "malloc(", "free(",
        "char *", "int *", "void *", "struct ", "typedef ", "union ",
        "enum ", "sizeof", "NULL", "->", "FILE*", "fopen(", "fclose(",
        "fread(", "fwrite(", "fprintf(", "fscanf(", "return 0;",
    },
    "rust": {
        "fn ", "let mut", "let ", "-> ", "impl ", "struct ", "enum ",
        "trait ", "pub ", "use ", "mod ", "match ", "Some(", "None",
        "Ok(", "Err(", "unwrap()", "expect(", "Result", "Option",
        "Vec::new()", "HashMap::new()", "String::from", "println!",
        "format!", "vec!", "loop ", "while let", "if let", "&mut",
        "&self", "&str", "&[u8]", "Box::new", "Rc::new", "Arc::new",
        "dyn ", "impl ", "where ", "for <", "lifetime", "'a",
        "#[derive", "#[test]", "#[cfg",
    },
    "go": {
        "func ", "package ", "import (", "fmt.", "Println", "Sprintf",
        "defer ", "go ", "chan ", "make(", "range ", "struct ", "interface ",
        "type ", "map[", "[]string", "[]int", "error", "nil", ":= ",
        "var ", "const ", "return ", "if ", "else ", "switch ", "case ",
        "default:", "break ", "continue ", "for ", "fallthrough",
        "goroutine", "select {", "context.", "http.", "json.",
    },
    "sql": {
        "SELECT ", "FROM ", "WHERE ", "INSERT INTO", "UPDATE ", "DELETE FROM",
        "CREATE TABLE", "ALTER TABLE", "DROP TABLE", "JOIN ", "LEFT JOIN",
        "RIGHT JOIN", "INNER JOIN", "GROUP BY", "ORDER BY", "HAVING",
        "LIMIT ", "OFFSET ", "UNION ", "DISTINCT ", "COUNT(", "SUM(",
        "AVG(", "MIN(", "MAX(", "LIKE ", "IN (", "BETWEEN ", "IS NULL",
        "IS NOT NULL", "AND ", "OR ", "NOT ", "PRIMARY KEY", "FOREIGN KEY",
        "INDEX ", "TRIGGER ", "VIEW ", "BEGIN", "COMMIT", "ROLLBACK",
    },
    "bash": {
        "#!/bin", "echo ", "if ", "then", "elif", "else", "fi", "for ",
        "do", "done", "while ", "until ", "case ", "esac", "function ",
        "export ", "local ", "source ", ". ", "exit ", "return ",
        "$(", "${", "`", "| ", "> ", "< ", ">> ", "2>&1", "&",
    },
    "php": {
        "<?php", "echo ", "function ", "public function", "private ",
        "protected ", "static ", "$this->", "self::", "parent::",
        "new ", "class ", "interface ", "abstract ", "trait ", "namespace ",
        "use ", "require_once", "include_once", "array(", "=> ",
        "foreach", "as ", "try {", "catch (", "throw ", "finally ",
        "->", "::", "null", "true", "false", "isset()", "empty()",
    },
    "swift": {
        "import UIKit", "import SwiftUI", "var ", "let ", "func ",
        "class ", "struct ", "enum ", "protocol ", "extension ",
        "override ", "init(", "self.", "super.", "nil", "optional",
        "guard ", "defer ", "throw ", "throws ", "rethrows", "-> ",
        "in ", "some ", "any ", "@State", "@Binding", "@ObservedObject",
        "@Published", "@Environment", "@main", "struct ContentView",
        "VStack", "HStack", "ZStack", "NavigationView", "List {",
    },
    "kotlin": {
        "fun ", "val ", "var ", "class ", "object ", "companion ",
        "data class", "sealed class", "enum class", "interface ",
        "override ", "open ", "abstract ", "private ", "internal ",
        "protected ", "public ", "constructor", "init {", "by ",
        "lazy", "lateinit", "?.", "?:", "!!", "===", "is ", "as ",
        "when ", "in ", "..", "until", "step", "downTo",
        "suspend ", "coroutine", "launch ", "async ",
        "viewModel", "Android", "@Composable", "remember ",
    },
    "ruby": {
        "def ", "class ", "module ", "end", "puts ", "print ",
        "attr_accessor", "attr_reader", "attr_writer", "initialize",
        "self.", "@", "@@", "$", "nil", "true", "false",
        "each ", "map ", "select ", "reject ", "reduce ",
        "do |", "{ |", "->", "lambda", "Proc.new",
        "raise ", "rescue ", "ensure ", "begin", "private ",
        "protected ", "public ", "include ", "extend ", "prepend ",
        "yield", "block_given?", "||=",
    },
    "scala": {
        "object ", "class ", "trait ", "def ", "val ", "var ",
        "extends ", "with ", "implicit ", "match {", "case ",
        "=>", "<-", "::", "Nil", "Option", "Some", "None",
        "Future", "Promise", "ExecutionContext", "map ",
        "flatMap ", "filter ", "foreach ", "fold ", "reduce ",
        "import scala.", "sealed ", "abstract ", "override ",
        "type ", "new ", "null", "Unit", "Any", "Nothing",
    },
    "html": {
        "<!DOCTYPE", "<html", "<head>", "<body>", "<div>", "<p>",
        "<h1", "<h2", "<h3", "<a ", "<img ", "<ul>", "<ol>", "<li>",
        "<table>", "<tr>", "<td>", "<th>", "<form>", "<input",
        "<button", "<select>", "<textarea>", "<label>", "<span>",
        "<section>", "<article>", "<header>", "<footer>", "<nav>",
        "<main>", "<aside>", "class=", "id=", "href=", "src=",
        "style=", "<!--", "</", "<script", "<link ", "<meta ",
    },
    "css": {
        "margin:", "padding:", "color:", "background:", "font-",
        "display:", "position:", "flex", "grid", "border:",
        "width:", "height:", "max-width", "min-height", "overflow:",
        "z-index", "opacity:", "transform:", "transition:", "animation:",
        "@media", "@keyframes", "@import", ":hover", ":focus",
        ":before", ":after", ":nth-child", ":first-child", ":last-child",
        ".class", "#id", "!important", "calc(", "var(--",
    },
    "r": {
        "library(", "require(", "function(", "<-", "->", "$",
        "data.frame", "list(", "c(", "mean(", "sd(", "sum(",
        "summary(", "head(", "tail(", "plot(", "ggplot(", "aes(",
        "geom_", "theme_", "labs(", "filter(", "select(", "mutate(",
        "group_by", "summarise", "arrange(", "ggplot2", "tidyverse",
        "lm(", "glm(", "predict(", "summary(", "print(", "cat(",
    },
    "matlab": {
        "function ", "end", "for ", "while ", "if ", "elseif ",
        "plot(", "figure", "xlabel", "ylabel", "title", "legend",
        "hold on", "grid on", "axis ", "subplot(", "colormap",
        "zeros(", "ones(", "eye(", "rand(", "size(", "length(",
        "reshape(", "repmat(", "linspace", "logspace",
        "fft(", "ifft(", "conv(", "filter(", "polyfit",
        "syms ", "diff(", "int(", "solve(", "dsolve(",
        "disp(", "fprintf(", "input(", "eval(", "feval(",
        "% ", "%% ", "matrix", "vector",
    },
}


class KnowledgeBase:
    """Stores structured text facts with TF-IDF semantic search, programming
    language detection, code pattern extraction, and learning validation.

    Search behaviour:
        1. TF-IDF cosine similarity (pure-Python, no deps) — always available.
        2. scikit-learn ``TfidfVectorizer`` — used when sklearn is installed
           (faster and more feature-rich).
        3. sentence-transformers embeddings — used when ``sentence_transformers``
           is installed (deep semantic understanding).
        4. Falls back to word-overlap if no index has been built yet.
    """

    def __init__(self) -> None:
        self.facts: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        # Programming-specific storage
        self.code_patterns: Dict[str, List[str]] = {}
        self.concepts: Dict[str, Set[str]] = {}

        # Learning validation tracking
        self.learning_scores: List[float] = []
        self.accuracy_history: List[float] = []

        # --- Semantic search index (lazy-built) ---
        self._tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self._fact_vectors: List[List[float]] = []
        self._sklearn_vectorizer: Any = None  # sklearn TfidfVectorizer if available
        self._sklearn_fact_vectors: Any = None  # sparse matrix from sklearn
        self._st_model: Any = None  # sentence-transformers model
        self._st_embeddings: Any = None  # numpy array of embeddings
        self._index_dirty: bool = True  # set True when facts change

    # ------------------------------------------------------------------
    # Fact management
    # ------------------------------------------------------------------

    def add_fact(self, text: str, meta: Optional[Dict[str, Any]] = None) -> None:
        """Add a factual text snippet to the knowledge base with automatic
        language detection and index invalidation."""
        if not text or not text.strip():
            return
        text = text.strip()
        self.facts.append(text)
        meta = meta or {}

        lang = self._detect_language(text)
        if lang:
            meta["language"] = lang
        if lang and self._contains_code(text):
            if lang not in self.code_patterns:
                self.code_patterns[lang] = []
            extracted = self._extract_code_patterns(text, lang)
            existing = set(self.code_patterns[lang])
            for pat in extracted:
                if pat not in existing:
                    self.code_patterns[lang].append(pat)
                    existing.add(pat)

        self.metadata.append(meta)
        self._index_dirty = True  # new fact → rebuild needed

    def add_facts(self, texts: List[str]) -> None:
        for t in texts:
            self.add_fact(t)

    def clear(self) -> None:
        self.facts.clear()
        self.metadata.clear()
        self.code_patterns.clear()
        self.concepts.clear()
        self._index_dirty = False
        self._tfidf_vectorizer = None
        self._fact_vectors = []
        self._sklearn_vectorizer = None
        self._sklearn_fact_vectors = None
        self._st_model = None
        self._st_embeddings = None

    def __len__(self) -> int:
        return len(self.facts)

    # ------------------------------------------------------------------
    # Semantic search — the core improvement
    # ------------------------------------------------------------------

    def rebuild_index(self) -> None:
        """(Re-)build the semantic search index from all current facts.

        Tries, in order of preference:
          1. sentence-transformers (deep semantic embeddings)
          2. scikit-learn TfidfVectorizer
          3. pure-Python TfidfVectorizer (always works)

        If the knowledge base is empty this is a no-op.
        """
        if not self.facts:
            self._index_dirty = False
            return

        # --- 1. Try sentence-transformers ---
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer

            if self._st_model is None:
                self._st_model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = self._st_model.encode(self.facts, show_progress_bar=False)
            self._st_embeddings = embeddings  # shape (n_facts, 384)
            self._tfidf_vectorizer = None
            self._sklearn_vectorizer = None
            self._index_dirty = False
            return
        except ImportError:
            self._st_model = None
            self._st_embeddings = None
        except Exception:
            self._st_model = None
            self._st_embeddings = None

        # --- 2. Try scikit-learn TfidfVectorizer ---
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf

            self._sklearn_vectorizer = SkTfidf(
                max_features=5000,
                stop_words="english",
                sublinear_tf=True,
            )
            matrix = self._sklearn_vectorizer.fit_transform(self.facts)
            self._sklearn_fact_vectors = matrix
            self._tfidf_vectorizer = None
            self._index_dirty = False
            return
        except ImportError:
            self._sklearn_vectorizer = None
            self._sklearn_fact_vectors = None
        except Exception:
            self._sklearn_vectorizer = None
            self._sklearn_fact_vectors = None

        # --- 3. Pure-Python TF-IDF (always works) ---
        vec = TfidfVectorizer(max_features=5000)
        vectors = vec.fit_transform(self.facts)
        self._tfidf_vectorizer = vec
        self._fact_vectors = vectors
        self._index_dirty = False

    def search(self, query: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Semantic search using TF-IDF / sentence-transformers with graceful
        fallback to word-overlap.

        Args:
            query: Search string.
            top_k: Maximum number of results.

        Returns:
            List of ``(fact_text, relevance_score)`` sorted by relevance.
        """
        if not self.facts or not query:
            return []

        # Ensure index is up-to-date
        if self._index_dirty:
            self.rebuild_index()

        # --- Try semantic search -------------------------------------------------
        results = self._semantic_search(query, top_k)
        if results:
            return results

        # --- Fallback: word-overlap search (always works) ------------------------
        return self._word_overlap_search(query, top_k)

    def _semantic_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Run semantic search via whichever index is available.

        Returns [] if no index is built.
        """
        if not self.facts:
            return []

        try:
            # --- sentence-transformers path ---
            if self._st_embeddings is not None:
                import numpy as np

                q_vec = self._st_model.encode([query], show_progress_bar=False)[0]
                scores = np.dot(self._st_embeddings, q_vec) / (
                    np.linalg.norm(self._st_embeddings, axis=1) * np.linalg.norm(q_vec) + 1e-12
                )
                top_indices = np.argsort(scores)[-top_k:][::-1]
                return [
                    (self.facts[i], float(scores[i]))
                    for i in top_indices
                    if float(scores[i]) > 0.0
                ]

            # --- scikit-learn TF-IDF path ---
            if self._sklearn_vectorizer is not None and self._sklearn_fact_vectors is not None:
                import numpy as np

                q_vec = self._sklearn_vectorizer.transform([query])
                scores = (self._sklearn_fact_vectors @ q_vec.T).toarray().flatten()
                top_indices = np.argsort(scores)[-top_k:][::-1]
                return [
                    (self.facts[i], float(scores[i]))
                    for i in top_indices
                    if float(scores[i]) > 0.0
                ]

            # --- Pure-Python TF-IDF path ---
            if self._tfidf_vectorizer is not None and self._fact_vectors:
                q_vecs = self._tfidf_vectorizer.transform([query])
                if q_vecs:
                    q_vec = q_vecs[0]
                    scored = []
                    for idx, fv in enumerate(self._fact_vectors):
                        sim = cosine_similarity(q_vec, fv)
                        if sim > 0.0:
                            scored.append((self.facts[idx], sim))
                    scored.sort(key=lambda x: -x[1])
                    return scored[:top_k]

        except Exception:
            pass

        return []

    def _word_overlap_search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Original word-overlap fallback."""
        query_lower = query.lower()
        query_words = set(re.findall(r"\w+", query_lower))
        if not query_words:
            return []

        scores: List[Tuple[str, float]] = []
        code_terms = {"function", "class", "method", "variable", "loop",
                      "array", "object", "string", "import", "return",
                      "def", "const", "let", "var", "type", "interface"}
        key_constructs = {"for", "while", "if", "else", "try", "catch",
                          "with", "as", "lambda", "yield", "async", "await"}

        for fact in self.facts:
            fact_lower = fact.lower()
            fact_words = re.findall(r"\w+", fact_lower)
            if not fact_words:
                continue

            overlap = sum(1 for w in fact_words if w in query_words)
            base_score = overlap / (math.sqrt(len(fact_words)) + 1e-5)

            # Language-match boost
            q_lang = self._detect_language(query)
            f_lang = self._detect_language(fact)
            if q_lang and f_lang and q_lang == f_lang:
                base_score *= 1.5

            code_overlap = len(query_words & code_terms & set(fact_words))
            if code_overlap:
                base_score *= 1.0 + 0.2 * code_overlap

            construct_overlap = len(query_words & key_constructs & set(fact_words))
            if construct_overlap:
                base_score *= 1.0 + 0.3 * construct_overlap

            if base_score > 0:
                scores.append((fact, base_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def search_by_language(self, query: str, language: str, top_k: int = 3) -> List[Tuple[str, float]]:
        """Filter search to facts detected as ``language``.

        Uses word-overlap scoring on the language subset because the
        TF-IDF / sentence-transformers index was built for *all* facts
        and would produce incorrect indices on a subset.
        """
        lang_facts = self.get_facts_by_language(language)
        if not lang_facts:
            return []
        query_words = set(re.findall(r"\w+", query.lower()))
        if not query_words:
            return []
        scores: List[Tuple[str, float]] = []
        for fact in lang_facts:
            fact_words = re.findall(r"\w+", fact.lower())
            if not fact_words:
                continue
            overlap = sum(1 for w in fact_words if w in query_words)
            score = overlap / (math.sqrt(len(fact_words)) + 1e-5)
            if score > 0:
                scores.append((fact, score))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ------------------------------------------------------------------
    # Language detection helpers (unchanged from previous version)
    # ------------------------------------------------------------------

    def _detect_language(self, text: str) -> Optional[str]:
        if not text:
            return None
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for lang, keywords in PROGRAMMING_LANGUAGES.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[lang] = score
        if not scores:
            return None
        return max(scores, key=scores.get) if max(scores.values()) >= 2 else None

    def _contains_code(self, text: str) -> bool:
        code_indicators = [
            r"\{", r"\}", r"\(", r"\)", r";$", r"def\s+\w+\s*\(",
            r"function\s+\w+\s*\(", r"class\s+\w+", r"fn\s+\w+",
            r"func\s+\w+", r"#include", r"import\s+\w+", r"SELECT",
            r"^\s+\w+", r"=>", r"::", r"->", r"\.\w+\(",
        ]
        return any(re.search(p, text, re.MULTILINE) for p in code_indicators)

    def _extract_code_patterns(self, text: str, language: str) -> List[str]:
        patterns = []
        func_patterns = {
            "python": r"def\s+\w+\s*\([^)]*\)\s*(->\s*\w+)?:",
            "javascript": r"(?:async\s+)?function\s+\w+\s*\([^)]*\)\s*\{?",
            "typescript": r"(?:async\s+)?function\s+\w+\s*\([^)]*\):\s*\w+\s*\{?",
            "java": r"(?:public|private|protected)\s+(?:static\s+)?\w+\s+\w+\s*\([^)]*\)",
            "cpp": r"\w+\s+\w+\s*\([^)]*\)\s*(?:const\s*)?\{?",
            "rust": r"fn\s+\w+\s*\([^)]*\)\s*(?:->\s*\w+)?\s*\{?",
            "go": r"func\s+(?:\(\s*\w+\s+\*\w+\s*\))?\s*\w+\s*\([^)]*\)\s*(?:\w+\s*)?\{?",
        }
        pat = func_patterns.get(language)
        if pat:
            patterns.extend(re.findall(pat, text, re.MULTILINE))
        class_patterns = {
            "python": r"class\s+\w+[\s\S]{0,50}?:",
            "java": r"(?:public\s+)?(?:abstract\s+)?class\s+\w+",
            "cpp": r"class\s+\w+[\s\S]{0,30}?\{",
            "rust": r"(?:pub\s+)?(?:struct|enum|trait)\s+\w+",
            "javascript": r"class\s+\w+",
        }
        pat2 = class_patterns.get(language)
        if pat2:
            patterns.extend(re.findall(pat2, text, re.MULTILINE))
        return patterns

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_programming_languages(self) -> List[str]:
        langs: Set[str] = set()
        for meta in self.metadata:
            lang = meta.get("language")
            if lang:
                langs.add(lang)
        return sorted(langs)

    def get_facts_by_language(self, language: str) -> List[str]:
        return [
            fact for fact, meta in zip(self.facts, self.metadata)
            if meta.get("language") == language
        ]

    def get_facts_by_category(self, category: str) -> List[str]:
        """Get facts tagged with a specific metadata category tag."""
        return [
            fact for fact, meta in zip(self.facts, self.metadata)
            if meta.get("category") == category
        ]

    def has_code_knowledge(self) -> bool:
        return any(meta.get("language") is not None for meta in self.metadata)

    def load_grammar_knowledge(self) -> int:
        """Load all ENGLISH_GRAMMAR entries as knowledge facts with
        'grammar' category metadata.

        Returns the number of grammar facts loaded.
        """
        count = 0
        for category, entries in ENGLISH_GRAMMAR.items():
            for entry in entries:
                self.add_fact(entry, meta={"category": "grammar", "grammar_topic": category})
                count += 1
        return count

    def search_grammar(self, query: str, topic: Optional[str] = None, top_k: int = 3) -> List[Tuple[str, float]]:
        """Search only grammar-related knowledge facts.

        Args:
            query: Search query string.
            topic: Optional grammar topic filter (e.g. 'tenses', 'parts_of_speech').
            top_k: Maximum number of results.

        Returns:
            List of (fact_text, relevance_score) tuples.
        """
        if topic:
            # Filter grammar facts by specific topic
            topic_facts = [
                f for f, m in zip(self.facts, self.metadata)
                if m.get("category") == "grammar" and m.get("grammar_topic") == topic
            ]
            if not topic_facts:
                return []
            saved = self.facts
            try:
                self.facts = topic_facts
                return self._word_overlap_search(query, top_k)
            finally:
                self.facts = saved

        # Search across all grammar facts
        grammar_facts = self.get_facts_by_category("grammar")
        if not grammar_facts:
            return []
        saved = self.facts
        try:
            self.facts = grammar_facts
            return self._word_overlap_search(query, top_k)
        finally:
            self.facts = saved

    # ------------------------------------------------------------------
    # Learning validation
    # ------------------------------------------------------------------

    def record_learning_score(self, score: float) -> None:
        self.learning_scores.append(max(0.0, min(1.0, score)))

    def record_accuracy(self, accuracy: float) -> None:
        self.accuracy_history.append(max(0.0, min(1.0, accuracy)))

    def get_average_learning_score(self) -> float:
        if not self.learning_scores:
            return 0.0
        return sum(self.learning_scores) / len(self.learning_scores)

    def get_average_accuracy(self) -> float:
        if not self.accuracy_history:
            return 0.0
        return sum(self.accuracy_history) / len(self.accuracy_history)

    def get_learning_status(self) -> Dict[str, Any]:
        return {
            "total_facts": len(self.facts),
            "code_languages": self.get_programming_languages(),
            "has_code_knowledge": self.has_code_knowledge(),
            "average_learning_score": self.get_average_learning_score(),
            "average_accuracy": self.get_average_accuracy(),
            "learning_checks_performed": len(self.learning_scores),
            "accuracy_checks_performed": len(self.accuracy_history),
        }


# ======================================================================
# LearningValidator (unchanged)
# ======================================================================

class LearningValidator:
    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.validation_history: List[Dict[str, Any]] = []

    def validate_learning(self, training_texts: List[str], generated_responses: List[str]) -> Dict[str, float]:
        if not training_texts or not generated_responses:
            return {"accuracy": 0.0, "comprehension_score": 0.0,
                    "knowledge_retention": 0.0, "language_understanding": 0.0}

        total_key_terms = 0
        retained_terms = 0
        for text in training_texts:
            key_terms = self._extract_key_terms(text)
            total_key_terms += len(key_terms)
            for term in key_terms:
                for response in generated_responses:
                    if term.lower() in response.lower():
                        retained_terms += 1
                        break

        retention = retained_terms / max(1, total_key_terms)

        correct_language = 0
        total_language_checks = 0
        for text, response in zip(training_texts[:len(generated_responses)], generated_responses):
            text_lang = self.knowledge_base._detect_language(text)
            resp_lang = self.knowledge_base._detect_language(response)
            if text_lang and resp_lang:
                total_language_checks += 1
                if text_lang == resp_lang:
                    correct_language += 1

        lang_understanding = correct_language / max(1, total_language_checks)
        comprehension = retention * 0.4 + lang_understanding * 0.6
        accuracy = retention * 0.3 + lang_understanding * 0.3 + comprehension * 0.4

        metrics = {
            "accuracy": round(accuracy, 4),
            "comprehension_score": round(comprehension, 4),
            "knowledge_retention": round(retention, 4),
            "language_understanding": round(lang_understanding, 4),
        }
        self.validation_history.append(metrics)
        self.knowledge_base.record_accuracy(accuracy)
        self.knowledge_base.record_learning_score(comprehension)
        return metrics

    def _extract_key_terms(self, text: str) -> Set[str]:
        """Extract key terms from text including technical, grammar, and general terms."""
        words = set(re.findall(r"\b[A-Z][a-z]+(?:\s+[a-z]+)*\b", text))

        # Programming and technical terms
        code_terms = set(re.findall(
            r"\b(def|class|function|import|return|var|let|const|if|else|for|while|try|catch|async|await|public|private|static|void|int|string|struct|enum|trait|impl|fn|func|package|interface|type|extends|implements)\b",
            text, re.IGNORECASE
        ))
        words.update(code_terms)

        # Grammar and language terms
        grammar_terms = set(re.findall(
            r"\b(noun|verb|adjective|adverb|pronoun|preposition|conjunction|determiner|article|tense|subject|object|clause|phrase|modifier|gerund|infinitive|participle|auxiliary|modal|plural|singular|possessive|relative|conditional|comparative|superlative)\b",
            text, re.IGNORECASE
        ))
        words.update(grammar_terms)

        return words

    def validate_grammar_understanding(self, text: str, response: str) -> float:
        """Validate that the model understands grammar concepts by checking
        for grammar terminology usage and sentence structure correctness."""
        score = 0.0
        checks = 0

        # Check sentence structure - sentences should end properly
        text_sentences = re.findall(r'[A-Z][^.!?]*[.!?]', text)
        resp_sentences = re.findall(r'[A-Z][^.!?]*[.!?]', response)
        if text_sentences and resp_sentences:
            checks += 1
            # Check if response uses proper sentence capitalization
            proper_start = sum(1 for s in resp_sentences if s[0].isupper())
            score += proper_start / len(resp_sentences)

        # Check grammar term usage
        grammar_terms = set(re.findall(
            r"\b(noun|verb|adjective|adverb|tense|plural|singular|subject|object|clause|phrase|sentence)\b",
            response, re.IGNORECASE
        ))
        if grammar_terms:
            checks += 1
            score += min(1.0, len(grammar_terms) / 3.0)

        return score / max(1, checks)

    def check_learning_convergence(self, threshold: float = 0.85, window: int = 5) -> bool:
        recent = self.validation_history[-window:]
        if len(recent) < window:
            return False
        avg_acc = sum(r["accuracy"] for r in recent) / len(recent)
        return avg_acc >= threshold

    def get_validation_summary(self) -> Dict[str, Any]:
        if not self.validation_history:
            return {"status": "no_validation_data", "runs": 0}
        avg_acc = sum(r["accuracy"] for r in self.validation_history) / len(self.validation_history)
        avg_comp = sum(r["comprehension_score"] for r in self.validation_history) / len(self.validation_history)
        avg_ret = sum(r["knowledge_retention"] for r in self.validation_history) / len(self.validation_history)
        avg_lang = sum(r["language_understanding"] for r in self.validation_history) / len(self.validation_history)
        return {
            "status": "converged" if self.check_learning_convergence() else "learning",
            "runs": len(self.validation_history),
            "average_accuracy": round(avg_acc, 4),
            "average_comprehension": round(avg_comp, 4),
            "average_retention": round(avg_ret, 4),
            "average_language_understanding": round(avg_lang, 4),
            "converged": self.check_learning_convergence(),
        }


# ======================================================================
# MemoryCache & ContextWindow (unchanged)
# ======================================================================

class MemoryCache:
    def __init__(self) -> None:
        self.cache: Dict[int, Tuple[Any, Any]] = {}

    def get(self, layer_idx: int) -> Optional[Tuple[Any, Any]]:
        return self.cache.get(layer_idx)

    def update(self, layer_idx: int, key: Any, value: Any) -> None:
        self.cache[layer_idx] = (key, value)

    def clear(self) -> None:
        self.cache.clear()


class ContextWindow:
    def __init__(self, max_tokens: int = 2048) -> None:
        self.max_tokens = max_tokens
        self.history: List[Dict[str, str]] = []

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.truncate_to_fit()

    def truncate_to_fit(self) -> None:
        total_chars = sum(len(m["content"]) for m in self.history)
        while total_chars > self.max_tokens * 4 and len(self.history) > 1:
            removed = self.history.pop(0)
            total_chars -= len(removed["content"])

    def get_prompt_text(self) -> str:
        formatted = []
        for msg in self.history:
            role = "User" if msg["role"] == "user" else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        return "\n".join(formatted)

    def clear(self) -> None:
        self.history.clear()
