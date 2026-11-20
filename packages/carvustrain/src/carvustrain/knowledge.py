"""Built-in educational knowledge corpora for CarvusTrain.

Provides ready-to-load facts for English grammar, mathematics, and science
so models start with useful world knowledge out of the box.
"""

from typing import Dict, List

# Re-export grammar from memory for a single knowledge entry point
from .memory import ENGLISH_GRAMMAR

MATH_KNOWLEDGE: Dict[str, List[str]] = {
    "arithmetic": [
        "Addition combines two or more numbers to find their sum: 3 + 5 = 8.",
        "Subtraction finds the difference between numbers: 10 - 4 = 6.",
        "Multiplication is repeated addition: 4 × 3 = 12.",
        "Division splits a number into equal parts: 12 ÷ 3 = 4.",
        "The order of operations is Parentheses, Exponents, Multiplication/Division, Addition/Subtraction (PEMDAS/BODMAS).",
        "A prime number is a natural number greater than 1 with only two factors: 1 and itself (2, 3, 5, 7, 11).",
        "Even numbers are divisible by 2; odd numbers leave a remainder of 1 when divided by 2.",
        "The absolute value |x| is the distance of x from zero on the number line.",
        "Fractions represent parts of a whole: 3/4 means three of four equal parts.",
        "Percentages express a number as a fraction of 100: 25% = 0.25 = 1/4.",
        "Decimals are base-10 representations of fractions: 0.5 = 1/2.",
        "The least common multiple (LCM) is the smallest positive integer divisible by each number in a set.",
        "The greatest common divisor (GCD) is the largest positive integer that divides each number in a set.",
    ],
    "algebra": [
        "A variable is a symbol (often x or y) that represents an unknown value.",
        "An equation states that two expressions are equal: 2x + 3 = 11.",
        "To solve a linear equation, isolate the variable using inverse operations.",
        "The slope-intercept form of a line is y = mx + b, where m is slope and b is y-intercept.",
        "Slope m = (y2 - y1) / (x2 - x1) measures the steepness of a line.",
        "A quadratic equation has the form ax² + bx + c = 0.",
        "The quadratic formula is x = (-b ± √(b² - 4ac)) / (2a).",
        "The discriminant Δ = b² - 4ac tells how many real roots a quadratic has.",
        "Exponents multiply a base by itself: aⁿ means a multiplied n times.",
        "Laws of exponents: aᵐ · aⁿ = aᵐ⁺ⁿ, (aᵐ)ⁿ = aᵐⁿ, aᵐ / aⁿ = aᵐ⁻ⁿ.",
        "A polynomial is a sum of terms like axⁿ where n is a non-negative integer.",
        "Factoring rewrites an expression as a product of simpler expressions.",
        "Inequalities use <, >, ≤, ≥ and reverse when multiplying/dividing by a negative.",
        "A system of equations can be solved by substitution, elimination, or graphing.",
        "Functions map each input x to exactly one output f(x).",
    ],
    "geometry": [
        "A point has position but no size; a line extends infinitely in both directions.",
        "An angle is formed by two rays sharing a common endpoint (vertex).",
        "A right angle measures 90 degrees; a straight angle measures 180 degrees.",
        "The sum of interior angles in a triangle is always 180 degrees.",
        "The Pythagorean theorem: in a right triangle, a² + b² = c² where c is the hypotenuse.",
        "Area of a rectangle = length × width; area of a triangle = (1/2) × base × height.",
        "Area of a circle = πr²; circumference = 2πr.",
        "Volume of a rectangular prism = length × width × height.",
        "Volume of a sphere = (4/3)πr³; volume of a cylinder = πr²h.",
        "Congruent shapes have the same size and shape; similar shapes have the same shape but may differ in size.",
        "Parallel lines never meet and have the same slope; perpendicular lines meet at 90 degrees.",
        "A polygon is a closed plane figure with straight sides.",
        "Pi (π) is the ratio of a circle's circumference to its diameter, approximately 3.14159.",
        "The distance between points (x1, y1) and (x2, y2) is √((x2-x1)² + (y2-y1)²).",
    ],
    "trigonometry": [
        "In a right triangle, sine = opposite / hypotenuse.",
        "In a right triangle, cosine = adjacent / hypotenuse.",
        "In a right triangle, tangent = opposite / adjacent.",
        "The reciprocal identities: cosecant = 1/sine, secant = 1/cosine, cotangent = 1/tangent.",
        "The Pythagorean identity: sin²θ + cos²θ = 1.",
        "Angles can be measured in degrees or radians; π radians = 180 degrees.",
        "The unit circle has radius 1 centered at the origin and defines trig values for all angles.",
        "Law of sines: a/sin(A) = b/sin(B) = c/sin(C).",
        "Law of cosines: c² = a² + b² - 2ab·cos(C).",
    ],
    "calculus_basics": [
        "A derivative measures how a function changes as its input changes (instantaneous rate of change).",
        "The derivative of xⁿ is n·xⁿ⁻¹ (power rule).",
        "An integral accumulates quantities and finds areas under curves.",
        "The fundamental theorem of calculus links differentiation and integration.",
        "A limit describes the value a function approaches as the input approaches a point.",
        "Continuity means a function has no jumps, holes, or breaks at a point.",
        "A local maximum or minimum occurs where the derivative is zero (critical points).",
    ],
    "statistics_probability": [
        "Mean is the average of a data set; median is the middle value; mode is the most frequent value.",
        "Range is the difference between the maximum and minimum values.",
        "Variance measures how far numbers are spread from their mean; standard deviation is its square root.",
        "Probability of an event is a number between 0 and 1 describing likelihood.",
        "Independent events: P(A and B) = P(A) · P(B).",
        "Mutually exclusive events cannot occur together: P(A or B) = P(A) + P(B).",
        "A normal distribution is a symmetric bell-shaped curve centered at the mean.",
        "Correlation measures linear association between two variables; causation is not implied by correlation alone.",
        "A sample is a subset of a population used to make inferences about the whole.",
        "Combinations count unordered selections; permutations count ordered arrangements.",
    ],
}

SCIENCE_KNOWLEDGE: Dict[str, List[str]] = {
    "physics": [
        "Newton's first law: an object at rest stays at rest, and an object in motion stays in motion, unless acted on by a net force.",
        "Newton's second law: Force = mass × acceleration (F = ma).",
        "Newton's third law: for every action there is an equal and opposite reaction.",
        "Energy cannot be created or destroyed, only converted (conservation of energy).",
        "Kinetic energy KE = (1/2)mv²; potential energy PE = mgh near Earth's surface.",
        "Work equals force times distance in the direction of the force: W = F·d.",
        "Power is the rate of doing work: P = W/t.",
        "Speed is distance over time; velocity includes direction.",
        "Acceleration is the rate of change of velocity.",
        "Gravity on Earth accelerates objects at approximately 9.8 m/s².",
        "Light travels at about 3 × 10⁸ meters per second in a vacuum.",
        "Sound travels through a medium as mechanical waves; it cannot travel in a vacuum.",
        "Ohm's law: Voltage = Current × Resistance (V = IR).",
        "Electric current is the flow of electric charge, measured in amperes.",
        "Waves transfer energy; frequency is waves per second, wavelength is distance between peaks.",
        "The electromagnetic spectrum includes radio, microwave, infrared, visible light, ultraviolet, X-rays, and gamma rays.",
        "Temperature measures average kinetic energy of particles; heat is energy transfer due to temperature difference.",
    ],
    "chemistry": [
        "An atom consists of a nucleus (protons and neutrons) surrounded by electrons.",
        "The atomic number equals the number of protons and identifies the element.",
        "The periodic table organizes elements by atomic number and chemical properties.",
        "A molecule is two or more atoms chemically bonded together.",
        "A chemical compound contains atoms of different elements bonded in fixed ratios.",
        "Ionic bonds form by electron transfer; covalent bonds form by electron sharing.",
        "The pH scale measures acidity: below 7 is acidic, 7 is neutral, above 7 is basic.",
        "A chemical reaction rearranges atoms; reactants form products.",
        "Conservation of mass: matter is neither created nor destroyed in ordinary chemical reactions.",
        "States of matter include solid, liquid, gas, and plasma.",
        "Water (H₂O) is a polar molecule essential for life.",
        "Acids donate protons (H⁺); bases accept protons or donate OH⁻.",
        "The mole is 6.022 × 10²³ particles (Avogadro's number).",
        "Oxidation is loss of electrons; reduction is gain of electrons (OIL RIG).",
        "Catalysts speed up reactions without being consumed.",
        "Solutions are homogeneous mixtures of solute dissolved in solvent.",
    ],
    "biology": [
        "All living organisms are made of one or more cells (cell theory).",
        "DNA (deoxyribonucleic acid) stores genetic information as sequences of A, T, C, and G bases.",
        "Genes are segments of DNA that code for proteins or functional RNA.",
        "Photosynthesis converts light energy into chemical energy: CO₂ + H₂O → sugar + O₂.",
        "Cellular respiration releases energy from food: sugar + O₂ → CO₂ + H₂O + ATP.",
        "Evolution by natural selection explains how populations change over generations.",
        "Ecosystems include living organisms interacting with their physical environment.",
        "Prokaryotes lack a nucleus; eukaryotes have a nucleus and organelles.",
        "Mitosis produces two identical diploid cells; meiosis produces four haploid gametes.",
        "Proteins are polymers of amino acids that perform most cellular functions.",
        "Enzymes are biological catalysts that speed up biochemical reactions.",
        "Homeostasis is the maintenance of stable internal conditions.",
        "The nervous system transmits signals; the endocrine system uses hormones.",
        "Viruses are infectious agents that require host cells to replicate.",
        "Biodiversity is the variety of life forms in an ecosystem or on Earth.",
        "Classification groups organisms into domain, kingdom, phylum, class, order, family, genus, species.",
    ],
    "earth_space": [
        "Earth orbits the Sun once per year and rotates on its axis once per day.",
        "The Moon orbits Earth and causes tides through gravitational pull.",
        "The solar system includes the Sun, eight planets, dwarf planets, moons, asteroids, and comets.",
        "A star produces energy by nuclear fusion of hydrogen into helium in its core.",
        "Galaxies are vast systems of stars, gas, and dust; the Milky Way is our galaxy.",
        "The Big Bang theory describes the origin and expansion of the universe.",
        "Plate tectonics explains earthquakes, volcanoes, and mountain formation.",
        "The water cycle moves water through evaporation, condensation, precipitation, and runoff.",
        "Weather is short-term atmospheric conditions; climate is long-term average weather.",
        "Fossil fuels (coal, oil, natural gas) formed from ancient organic matter.",
        "The atmosphere is mostly nitrogen (~78%) and oxygen (~21%).",
        "Earth's magnetic field protects the surface from solar wind.",
    ],
    "scientific_method": [
        "Science relies on observation, hypothesis, experiment, analysis, and conclusion.",
        "A hypothesis is a testable proposed explanation for an observation.",
        "A controlled experiment changes one independent variable while keeping others constant.",
        "Peer review and reproducibility strengthen scientific claims.",
        "A theory in science is a well-supported explanatory framework, not a guess.",
        "Correlation does not prove causation; experiments help establish causal links.",
        "Models simplify complex systems to make predictions and explanations.",
        "Units should be SI (metric): meter, kilogram, second, ampere, kelvin, mole, candela.",
    ],
}


def flatten_corpus(corpus: Dict[str, List[str]]) -> List[str]:
    """Flatten a topic→facts mapping into a single list of fact strings."""
    facts: List[str] = []
    for entries in corpus.values():
        facts.extend(entries)
    return facts


def all_builtin_topics() -> Dict[str, int]:
    """Return topic name → fact count for all built-in knowledge corpora."""
    return {
        "grammar": sum(len(v) for v in ENGLISH_GRAMMAR.values()),
        "math": sum(len(v) for v in MATH_KNOWLEDGE.values()),
        "science": sum(len(v) for v in SCIENCE_KNOWLEDGE.values()),
    }
