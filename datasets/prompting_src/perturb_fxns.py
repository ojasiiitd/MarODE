import random
import re

# -----------------------
# Perturbation functions (generalized)
# -----------------------
def perturb_ordering(steps):
    if len(steps) > 2:
        i, j = random.sample(range(len(steps)-1), 2)
        if i != j:  # Only swap if different indices
            steps[i], steps[j] = steps[j], steps[i]
            return True
    return False

def perturb_deletion(steps):
    if len(steps) > 1:
        idx = random.randrange(len(steps))
        steps.pop(idx)
        return True
    return False

def perturb_duplication(steps):
    if steps:
        idx = random.randrange(len(steps))
        steps.insert(idx, steps[idx])
        return True
    return False

def perturb_negation(steps):
    if not steps:
        return False

    NEGATION_TRIGGERS = [
        "is","are","was","were","has","have","can","could","shall","should",
        "does","do","did","will","would","may","might","must","always","often"
    ]

    # pick a random step but only from those that contain a trigger
    candidate_indices = [
        i for i, s in enumerate(steps)
        if any(t in s.lower().split() for t in NEGATION_TRIGGERS)
    ]
    if not candidate_indices:
        return False

    idx = random.choice(candidate_indices)
    words = steps[idx].split()
    for i, w in enumerate(words):
        if w.lower() in NEGATION_TRIGGERS:
            words.insert(i+1, "not")
            steps[idx] = " ".join(words)
            return True

    return False

HEDGE_MAP = {
    " is ": " might be ",
    " are ": " could be ",
    " will ": " may ",
    " causes ": " may cause ",
    " therefore ": " perhaps ",
    " always ": " often ",
    " never ": " rarely ",
    " must ": " may ",
    " should ": " might ",
    " can ": " could ",
    " proves ": " suggests ",
    " shows ": " appears to show ",
    " results in ": " can result in ",
    " indicates ": " seems to indicate ",
    " demonstrates ": " seems to demonstrate ",
    " leads to ": " may lead to ",
    " clearly ": " arguably ",
    " definitely ": " possibly "
}
def perturb_hedge(steps):
    if steps:
        idx = random.randrange(len(steps))
        original_step = steps[idx]
        for k, v in HEDGE_MAP.items():
            if k in steps[idx]:
                steps[idx] = steps[idx].replace(k, v)
                # Return True only if the step actually changed
                if steps[idx] != original_step:
                    return True
                else:
                    steps[idx] = original_step  # Restore original
    return False

def perturb_number(steps):
    # possible changes: +1, -1, +10, -10, +100, -100
    deltas = [1, 10, 100]

    for i, s in enumerate(steps):
        nums = re.findall(r"\d+", s)
        if nums:
            n = int(nums[0])
            delta = random.choice(deltas)           # 1, 10, or 100
            if random.choice([True, False]):        # randomly choose + or -
                new_n = n + delta
            else:
                new_n = n - delta
            # build new string with first occurrence replaced
            new_s = s.replace(str(n), str(new_n), 1)
            if new_s != s:  # Only if replacement happened
                steps[i] = new_s
                return True
    return False

def perturb_entity(steps):
    all_caps = [w for s in steps for w in s.split() if w.istitle() and len(w) > 1]
    if len(all_caps) >= 2:
        src, tgt = random.sample(all_caps, 2)
        idx = random.randrange(len(steps))
        if src in steps[idx]:
            new_step = steps[idx].replace(src, tgt)
            if new_step != steps[idx]:  # Only if replacement occurred
                steps[idx] = new_step
                return True
    return False

PRONOUNS = {
    "he": "she",
    "she": "he",
    "it": "they",
    "they": "it",
    "his": "her",
    "her": "his",
    "him": "her",
    "hers": "his",
    "we": "I",
    "i": "we",
    "you": "they",
    "us": "them",
    "them": "us"
}

def perturb_pronoun(steps):
    """Replace a pronoun in one step with another pronoun."""
    if not steps:
        return False

    # find all candidate indices where at least one pronoun appears
    candidate_indices = []
    for i, s in enumerate(steps):
        if any(re.search(rf"\b{p}\b", s, flags=re.I) for p in PRONOUNS):
            candidate_indices.append(i)

    if not candidate_indices:
        return False

    idx = random.choice(candidate_indices)
    text = steps[idx]

    # pick a random pronoun from the map that actually occurs in this step
    pronouns_in_step = [p for p in PRONOUNS if re.search(rf"\b{p}\b", text, flags=re.I)]
    if not pronouns_in_step:
        return False

    k = random.choice(pronouns_in_step)
    v = PRONOUNS[k]

    new_text = re.sub(rf"\b{k}\b", v, text, count=1, flags=re.I)
    if new_text != text:
        steps[idx] = new_text
        return True

    return False

TEMPLATES = [
    "It is known that",
    "In general,",
    "It is often said that",
    "Researchers suggest that",
    "Studies have shown that",
    "Experts believe that",
    "It has been observed that",
    "Evidence indicates that",
    "It can be argued that",
    "According to many,",
    "Some argue that",
    "It is widely accepted that",
    "Traditionally,",
    "It is reported that",
    "It is believed that",
    "Observations reveal that",
    "It appears that",
    "It is understood that",
    "In many cases,",
    "It is frequently noted that"
]
def perturb_template(steps):
    if steps:
        idx = random.randrange(len(steps))
        original_step = steps[idx]
        template = random.choice(TEMPLATES)
        steps[idx] = template + " " + steps[idx]
        # Only return True if the step actually changed (template wasn't already there)
        if steps[idx] != original_step and not original_step.startswith(template):
            return True
        else:
            steps[idx] = original_step  # Restore original
    return False

# -----------------------
# New 10 Worsening Perturbations
# -----------------------

def perturb_circular(steps):
    if steps:
        idx = random.randrange(len(steps))

        # list of possible prefixes
        prefixes = [
            "This is true because",
            "This is the case because",
            "This holds true since",
            "This can be explained by",
            "The evidence shows that",
            "As a result of"
        ]

        # only add if not already prefixed with one of them
        if not any(steps[idx].startswith(p) for p in prefixes):
            prefix = random.choice(prefixes)
            steps.insert(idx + 1, f"{prefix} {steps[idx]}")
            return True
    return False

def perturb_contradiction(steps):
    if not steps:
        return False

    idx = random.randrange(len(steps))
    original = steps[idx]

    # pool of contradiction phrases
    contradiction_phrases = [
        "However, the opposite is also true:",
        "Conversely, the reverse holds:",
        "On the other hand, an opposing view is:",
        "Paradoxically, the contrary is valid:",
        "In contrast, one could argue the opposite:"
    ]
    # only add if none of them already present
    if not any(phrase in original for phrase in contradiction_phrases):
        phrase = random.choice(contradiction_phrases)
        steps.insert(idx + 1, f"{phrase} {original}")
        return True

    return False

def perturb_redundant(steps):
    if not steps:
        return False
    # Exclude the last step from being chosen for redundancy
    if len(steps) > 1:
        idx = random.randrange(len(steps) - 1)
    else:
        # If there's only one step, we can apply redundancy to it
        idx = 0
    original_step = steps[idx]
    # Add the text of the same step to itself, separated by a space
    steps[idx] = f"{original_step} {original_step}"
    return True

def perturb_irrelevant_elaboration(steps):
    ELAB = [
        "People often eat breakfast in the morning.",
        "Many enjoy listening to the radio.",
        "Cats are popular pets.",
        "Shoes are worn on the feet.",
        "Water boils at 100 degrees Celsius at sea level.",
        "The sky can appear blue on a clear day.",
        "Most plants need sunlight to grow.",
        "Books are often kept on shelves.",
        "Birds have feathers and lay eggs.",
        "Computers use electricity to operate.",
        "Cars typically run on roads.",
        "Apples grow on trees.",
        "Winter is usually colder than summer.",
        "Rainbows have multiple colors.",
        "Mountains are higher than hills.",
        "Pens are used for writing.",
        "People sleep at night.",
        "Many countries have their own flags.",
        "Music can be played on instruments.",
        "Bicycles have two wheels."
    ]
    if steps:
        idx = random.randrange(len(steps))
        elaboration = random.choice(ELAB)
        # Only add if this elaboration isn't already in the steps
        if elaboration not in steps:
            steps.insert(idx + 1, elaboration)
            return True
    return False

def perturb_overgeneralization(steps):
    # High-impact overgeneralization patterns that work on most text
    GENERAL_PATTERNS = [
        # Replace moderate quantifiers with absolutes
        (r"\bsome\b", "all", re.IGNORECASE),
        (r"\bmany\b", "all", re.IGNORECASE),
        (r"\bseveral\b", "all", re.IGNORECASE),
        (r"\bfew\b", "none", re.IGNORECASE),
        (r"\bmost\b", "all", re.IGNORECASE),
        
        # Replace probabilistic terms with certainties
        (r"\boften\b", "always", re.IGNORECASE),
        (r"\bsometimes\b", "always", re.IGNORECASE),
        (r"\boccasionally\b", "always", re.IGNORECASE),
        (r"\busually\b", "always", re.IGNORECASE),
        (r"\brarely\b", "never", re.IGNORECASE),
        
        # Replace modal verbs with absolutes
        (r"\bcan\b", "must", re.IGNORECASE),
        (r"\bmay\b", "will", re.IGNORECASE),
        (r"\bmight\b", "will", re.IGNORECASE),
        (r"\bcould\b", "does", re.IGNORECASE),
        
        # Replace hedges with certainties
        (r"\bperhaps\b", "definitely", re.IGNORECASE),
        (r"\bpossibly\b", "certainly", re.IGNORECASE),
        (r"\bprobably\b", "certainly", re.IGNORECASE),
    ]
    
    # Try each pattern on each step
    for pattern, replacement, flags in GENERAL_PATTERNS:
        for i, step in enumerate(steps):
            if re.search(pattern, step, flags=flags):
                new_step = re.sub(pattern, replacement, step, flags=flags)
                if new_step != step:
                    steps[i] = new_step
                    return True
    
    # If no patterns matched, try more aggressive approaches
    for i, step in enumerate(steps):
        original_step = step
        
        # Add absolute modifiers to the beginning or end
        modifiers = [
            "Without any doubt, ",
            "In every case, ",
            "Universally, ",
            "Absolutely, ",
            "Without exception, ",
        ]
        
        # 50% chance to add at beginning, 50% at end
        if random.choice([True, False]):
            new_step = random.choice(modifiers) + step
        else:
            new_step = step + " without any exceptions."
        
        if new_step != original_step:
            steps[i] = new_step
            return True
    
    return False

AMBIGUOUS_TERMS = [
    "someone",
    "somebody",
    "some place",
    "somewhere",
    "a certain person",
    "a certain entity",
    "an unspecified individual",
    "an unnamed source",
    "a mysterious figure",
    "some entity",
    "an unknown party"
]
def perturb_underspecification(steps):
    """Replace proper nouns with a vague/ambiguous term."""
    for i, step in enumerate(steps):
        # only try if the step has a capitalised word (proper noun heuristic)
        if re.search(r"\b[A-Z][a-z]+\b", step):
            replacement = random.choice(AMBIGUOUS_TERMS)
            new_step = re.sub(r"\b[A-Z][a-z]+\b", replacement, step, count=1)
            if new_step != step:
                steps[i] = new_step
                return True
    return False

import random

TEMPORAL_STMTS = [
    "This actually happened in the 18th century.",
    "Surprisingly, this event took place during the Ice Age.",
    "This is believed to have occurred in the distant future.",
    "Historians claim it happened thousands of years ago.",
    "This phenomenon was first noted in prehistoric times.",
    "Curiously, records place this in the year 3025.",
    "This was reported to occur before written history.",
    "Evidence suggests this will happen in the next millennium.",
    "This is said to have occurred long before the pyramids were built.",
    "Some sources place this event after World War III."
]
def perturb_temporal_confusion(steps):
    if not steps:
        return False
    
    # Exclude the last step, which is usually the Final Verdict
    valid_indices = list(range(len(steps)))
    if steps[-1].strip().startswith("Final Verdict"):
        valid_indices = list(range(len(steps) - 1))

    if not valid_indices:
        return False
        
    idx = random.choice(valid_indices)
    temporal_stmt = random.choice(TEMPORAL_STMTS)
    
    # Check if the temporal statement is already a substring of the chosen step
    if temporal_stmt not in steps[idx]:
        # Add the temporal statement to the end of the chosen step's string
        steps[idx] += f" {temporal_stmt}"
        return True
    return False

def perturb_cause_effect_reversal(steps):
    '''
    Perturbs reasoning steps by reversing cause-effect relationships.
    
    This function attempts to identify and reverse causal relationships in reasoning steps
    using multiple strategies:
    1. Pattern-based reversal using explicit causal indicators (because, causes, leads to, etc.)
    2. Implicit causal verb reversal (increase, decrease, affect, etc.)
    3. Simple SVO (Subject-Verb-Object) pattern reversal
    
    Args:
        steps: List of reasoning steps to perturb
        
    Returns:
        bool: True if any perturbation was successfully applied, False otherwise
    '''
    # Expanded list of causal relationship indicators
    CAUSAL_PATTERNS = [
        (r'\bbecause\b', lambda parts: f"{parts[1].strip()} because {parts[0].strip()}"),
        (r'\bcauses\b', lambda step: step.replace("causes", "is caused by")),
        (r'\bleads to\b', lambda step: step.replace("leads to", "is a result of")),
        (r'\bresults in\b', lambda step: step.replace("results in", "stems from")),
        (r'\bproduces\b', lambda step: step.replace("produces", "is produced by")),
        (r'\bcreates\b', lambda step: step.replace("creates", "is created by")),
        (r'\bgenerates\b', lambda step: step.replace("generates", "is generated by")),
        (r'\btriggers\b', lambda step: step.replace("triggers", "is triggered by")),
        (r'\binduces\b', lambda step: step.replace("induces", "is induced by")),
        (r'\bpromotes\b', lambda step: step.replace("promotes", "is promoted by")),
        (r'\bstimulates\b', lambda step: step.replace("stimulates", "is stimulated by")),
        (r'\bhence\b', lambda step: step.replace("hence", "preceded by")),
        (r'\btherefore\b', lambda step: step.replace("therefore", "following from")),
        (r'\bthus\b', lambda step: step.replace("thus", "as a consequence of")),
        (r'\bconsequently\b', lambda step: step.replace("consequently", "due to the fact that")),
        (r'\bdue to\b', lambda step: step.replace("due to", "causes")),
        (r'\bowing to\b', lambda step: step.replace("owing to", "leads to")),
        (r'\bas a result\b', lambda step: step.replace("as a result", "as a cause")),
    ]
    
    for i, step in enumerate(steps):
        original_step = step
        
        # Try pattern-based reversals first
        for pattern, reversal_func in CAUSAL_PATTERNS:
            if re.search(pattern, step, flags=re.IGNORECASE):
                if pattern == r'\bbecause\b':
                    # Special handling for "because" which splits the sentence
                    parts = re.split(pattern, step, flags=re.IGNORECASE)
                    if len(parts) >= 2:
                        new_step = reversal_func(parts)
                        if new_step != original_step:
                            steps[i] = new_step
                            return True
                else:
                    # For other patterns, use the replacement function
                    new_step = reversal_func(step)
                    if new_step != original_step:
                        steps[i] = new_step
                        return True
        
        # Try to detect implicit causal relationships using common verbs
        IMPLICIT_CAUSAL_VERBS = [
            ('increase', 'is increased by'),
            ('decrease', 'is decreased by'),
            ('affect', 'is affected by'),
            ('influence', 'is influenced by'),
            ('change', 'is changed by'),
            ('modify', 'is modified by'),
            ('enhance', 'is enhanced by'),
            ('reduce', 'is reduced by'),
            ('improve', 'is improved by'),
            ('worsen', 'is worsened by')
        ]
        
        for verb, passive_form in IMPLICIT_CAUSAL_VERBS:
            # Look for patterns like "X verbs Y"
            pattern = rf'(\b[A-Za-z]+\b)\s+{verb}s?\s+(\b[A-Za-z]+\b)'
            match = re.search(pattern, step, flags=re.IGNORECASE)
            if match:
                subject, object_ = match.groups()
                new_step = re.sub(pattern, f"{object_} {passive_form} {subject}", step, flags=re.IGNORECASE)
                if new_step != original_step:
                    steps[i] = new_step
                    return True
        
        # Try SVO (Subject-Verb-Object) pattern reversal for simple sentences
        words = step.split()
        if len(words) >= 3 and words[0][0].isupper():  # Likely a sentence starting with capital
            # Simple pattern: Noun Verb Noun -> Noun is verbed by Noun
            if len(words) >= 3 and words[1].endswith('s'):  # Third person singular
                verb = words[1]
                if verb.endswith('s'):
                    base_verb = verb[:-1]
                    passive_verb = f"is {base_verb}ed by"
                    new_step = f"{words[2]} {passive_verb} {words[0]} {' '.join(words[3:])}".strip()
                    if new_step != original_step and len(new_step.split()) >= 3:
                        steps[i] = new_step
                        return True
    
    return False

def perturb_connector_abuse(steps):
    if not steps:
        return False

    idx = random.randrange(len(steps))
    original_step = steps[idx]

    # list of possible connector phrases
    connectors = [
        ", however this contradicts itself.",
        ", yet this undermines the original argument.",
        ", but this statement is inconsistent.",
        ", though it paradoxically refutes itself.",
        ", nevertheless this goes against the prior claim.",
        ", despite which it cancels itself out."
    ]

    # only add if not already containing any of them
    if not any(connector in original_step for connector in connectors):
        connector = random.choice(connectors)
        steps[idx] = original_step + connector
        return True

    return False

# def perturb_unsupported_conclusion(steps):
#     unsupported_conclusions = [
#         "It seems to suggest that unicorns exist.",
#         "This could imply that dragons are real.",
#         "One might take it as pointing to the moon being made of cheese.",
#         "The idea could be interpreted as cats being able to fly.",
#         "It gives the impression that time travel might be common.",
#         "It can be read as hinting that pigs might speak.",
#         "One possible takeaway is that rivers could flow uphill.",
#         "This might be taken as saying that trees walk at night.",
#         "It may be understood as suggesting the earth is flat.",
#         "A possible conclusion is that fish might breathe air like humans.",
#         "This reasoning appears to hint that rocks could have feelings.",
#         "It might be interpreted as claiming that shadows are alive.",
#         "From this, one could suppose that dogs might talk.",
#         "It can sound like clouds are made of cotton candy.",
#         "The pattern could point toward the sun revolving around the earth.",
#         "Such a view might mean that mirrors contain hidden worlds.",
#         "It may give the idea that stars are holes in the sky.",
#         "It can appear to suggest that mountains are hollow inside.",
#         "One could end up thinking that people might never need sleep.",
#         "It could be taken as evidence that rainbows act like bridges."
#     ]

#     if not steps:
#         return False  # nothing to perturb

#     conclusion = random.choice(unsupported_conclusions)
#     l_old = steps[-1]

#     # Add the conclusion to the last step (with a space in between if needed)
#     steps[-1] = steps[-1].rstrip() + " " + conclusion

#     # Check if modification happened
#     return steps[-1] != l_old

def perturb_unsupported_conclusion(steps, delete_ratio=0.3):
    """
    Perturb the last step by randomly deleting ~30% of its words.
    Updates the last step in place.
    Returns True if perturbation applied, False otherwise.
    """
    if not steps:
        return False  # nothing to perturb
    
    last_step = steps[-1].split()
    if len(last_step) < 2:
        return False  # skip very short steps
    
    # Decide how many words to delete (at least 1 if possible)
    num_to_delete = max(1, int(len(last_step) * delete_ratio))
    
    # Randomly choose positions to delete
    delete_indices = set(random.sample(range(len(last_step)), num_to_delete))
    
    perturbed_words = [w for i, w in enumerate(last_step) if i not in delete_indices]
    
    # Ensure at least one word remains
    if not perturbed_words:
        perturbed_words = [random.choice(last_step)]
    
    steps[-1] = " ".join(perturbed_words)
    return True    

VAGUE_QUANTIFIERS = [
    "a lot of", "many", "some people say", "a big", "a huge",
    "plenty of", "countless", "various", "numerous",
    "a significant number of", "lots of", "a fair amount of"
]
# patterns that match numbers, percentages, and common quantifiers
QUANTIFIER_PATTERNS = [
    r"\b\d+%?\b",                # numbers or percentages: 78 or 78%
    r"\b\d+\s*of\b",             # "3 of", "10 of"
    r"\b(one|two|three|four|five|six|seven|eight|nine|ten|hundred|thousand)\b",
    r"\b(most|majority|nearly all|almost all|several|some|few|numerous|multiple|various|hundreds)\b"
]
def perturb_quantifier_abuse(steps):
    """
    Replace precise quantities or scopes with vague quantifiers.
    """
    if not steps:
        return False

    # build list of indices that have any of the patterns
    candidate_indices = []
    for i, s in enumerate(steps):
        if any(re.search(pat, s, flags=re.IGNORECASE) for pat in QUANTIFIER_PATTERNS):
            candidate_indices.append(i)

    if not candidate_indices:
        # fallback: just pick a random step and prepend a vague quantifier
        idx = random.randrange(len(steps))
        steps[idx] = random.choice(VAGUE_QUANTIFIERS) + " " + steps[idx]
        return True

    idx = random.choice(candidate_indices)
    step = steps[idx]

    # pick a pattern that actually matches in this step
    matching_patterns = [pat for pat in QUANTIFIER_PATTERNS if re.search(pat, step, flags=re.IGNORECASE)]
    if not matching_patterns:
        return False

    pat = random.choice(matching_patterns)
    replacement = random.choice(VAGUE_QUANTIFIERS)

    # insert a trailing space if pattern ends with 'of'
    if pat.endswith("of\\b"):
        replacement = replacement + " "

    new_step = re.sub(pat, replacement, step, count=1, flags=re.IGNORECASE)
    if new_step != step:
        steps[idx] = new_step
        return True

    return False

def perturb_misplaced_modifier(steps):
    """
    Move a trailing adjectival/adverbial phrase (e.g. 'covered in X', 'with Y')
    from the end of a sentence to the beginning to create a misplaced modifier.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    step = steps[idx]

    # regex to catch common trailing modifier phrases
    # e.g. " covered in mustard and relish", " with great care"
    pattern = r"(,?\s*(covered in|with|using|including|containing|made of|due to|because of)[^\.!?]+)$"

    match = re.search(pattern, step, flags=re.IGNORECASE)
    if not match:
        return False

    phrase = match.group(1).strip()  # the trailing modifier phrase
    # remove the phrase from the end
    base = step[:match.start(1)].strip()

    # new step with the modifier phrase moved to the beginning
    # ensure proper spacing and punctuation
    new_step = f"{phrase.strip().rstrip(',')}, {base}"

    if new_step != step:
        steps[idx] = new_step
        return True

    return False

INTENSIFIERS = [
    "clearly",
    "indeed",
    "basically",
    "obviously",
    "plainly",
    "surely",
    "undoubtedly",
    "certainly",
    "truly",
    "frankly",
    "very",
    "absolutely"
]
def perturb_unjustified_emphasis(steps):
    """
    Insert an intensifier/discourse marker into a step to add unjustified emphasis.
    Example:
    'She is an expert in the field.' →
    'She is clearly an expert in the field.'
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    step = steps[idx].split()

    if len(step) < 2:
        return False  # too short to sensibly insert

    intensifier = random.choice(INTENSIFIERS)

    # choose an insertion point — after first or second word so it reads naturally
    insert_pos = random.choice([1, 2]) if len(step) > 2 else 1

    new_words = step[:insert_pos] + [intensifier] + step[insert_pos:]
    new_step = " ".join(new_words)

    if new_step != steps[idx]:
        steps[idx] = new_step
        return True

    return False

import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
def perturb_lexical_substitution(steps, prob_antonym=0.8):
    def get_synonyms(word):
        syns = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                syns.add(lemma.name().replace("_"," "))
        return list(syns)

    def get_antonyms(word):
        ants = set()
        for syn in wn.synsets(word):
            for lemma in syn.lemmas():
                if lemma.antonyms():
                    for a in lemma.antonyms():
                        ants.add(a.name().replace("_"," "))
        return list(ants)
    
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()

    # pick a candidate content word (skip very short function words)
    content_words = [w for w in words if len(w) > 3]
    if not content_words:
        return False

    w = random.choice(content_words)
    synonyms = get_synonyms(w)
    antonyms = get_antonyms(w)

    if random.random() < prob_antonym and antonyms:
        replacement = random.choice(antonyms)
    elif synonyms:
        replacement = random.choice(synonyms)
    else:
        return False  # nothing to replace

    # replace only first occurrence
    new_words = []
    replaced = False
    for token in words:
        if not replaced and token.lower() == w.lower():
            new_words.append(replacement)
            replaced = True
        else:
            new_words.append(token)

    new_step = " ".join(new_words)
    if new_step != steps[idx]:
        steps[idx] = new_step
        return True

    return False


def perturb_random_hyphenation(steps, hyphen_prob=0.5):
    """
    Randomly hyphenate multiple words (length ≥4) in one step.
    hyphen_prob controls the chance of hyphenating each eligible word.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    changed = False

    new_words = []
    for token in words:
        core = re.sub(r'^\W+|\W+$', '', token)  # strip punctuation
        if len(core) >= 4 and random.random() < hyphen_prob:
            pos = random.randint(1, len(core)-2)
            hyphenated = core[:pos] + "-" + core[pos:]
            new_token = token.replace(core, hyphenated, 1)
            new_words.append(new_token)
            changed = True
        else:
            new_words.append(token)

    if changed:
        steps[idx] = " ".join(new_words)
        return True

    return False


def perturb_word_truncation(steps):
    """
    Truncates a single word at a random position within a random step,
    leaving a dangling fragment.
    """
    # Return False if the list of steps is empty
    if not steps:
        return False

    # Select a random step from the list
    idx = random.randrange(len(steps))
    step = steps[idx]
    words = step.split()

    # Find all words long enough to be truncated (more than 1 character)
    long_word_indices = [i for i, word in enumerate(words) if len(word) > 1]

    # If no words can be truncated, the perturbation fails
    if not long_word_indices:
        return False

    # Choose a random word to truncate from the list of eligible words
    word_to_truncate_idx = random.choice(long_word_indices)
    word_to_truncate = words[word_to_truncate_idx]

    # Choose a random point to cut the word, ensuring at least one character is left
    # and at least one is removed.
    truncation_point = random.randint(1, len(word_to_truncate) - 1)
    
    # Create the truncated word
    truncated_word = word_to_truncate[:truncation_point]

    # Replace the original word with the truncated version
    words[word_to_truncate_idx] = truncated_word

    # Reassemble the step and update the list
    new_step = " ".join(words)
    steps[idx] = new_step
    
    return True

nltk.download('averaged_perceptron_tagger_eng')
def perturb_domain_shift(steps, replace_prob=0.8):
    """
    Replace multiple nouns in one step with other words from the same semantic domain
    (sharing a hypernym). replace_prob controls chance of replacement per noun.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)
    changed = False

    for i, (word, pos) in enumerate(tags):
        if pos.startswith('NN') and random.random() < replace_prob:
            synsets = wn.synsets(word, pos=wn.NOUN)
            if not synsets:
                continue

            # collect sister terms
            sister_terms = set()
            for syn in synsets:
                for hyper in syn.hypernyms():
                    for sister in hyper.hyponyms():
                        if sister != syn:
                            for lemma in sister.lemmas():
                                candidate = lemma.name().replace('_', ' ')
                                if candidate.lower() != word.lower():
                                    sister_terms.add(candidate)

            if sister_terms:
                replacement = random.choice(list(sister_terms))
                # preserve case style a little
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement
                changed = True

    if changed:
        steps[idx] = ' '.join(words)
        return True

    return False

def perturb_verb_aspect(steps, replace_prob=0.8):
    """
    Change verb forms to create temporal or aspectual confusion.
    Example: 'increases' -> 'increase', 'will increase' -> 'increases'.
    replace_prob controls how many verbs get perturbed per step.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)
    changed = False

    for i, (word, pos) in enumerate(tags):
        if pos.startswith('VB') and random.random() < replace_prob:
            # Very simple tense/aspect shifting:
            if word.lower().endswith('s') and len(word) > 2:
                # present 3rd person -> base form
                replacement = word[:-1]
            elif word.lower().endswith('ed') and len(word) > 3:
                # past tense -> add 's' to make it present third person
                replacement = word.rstrip('ed') + 's'
            else:
                # otherwise just add 's'
                replacement = word + 's'
            # preserve capitalization
            if word[0].isupper():
                replacement = replacement.capitalize()
            words[i] = replacement
            changed = True

    if changed:
        steps[idx] = ' '.join(words)
        return True

    return False


def perturb_adjective_intensity(steps, replace_prob=0.8):
    """
    Replace adjectives with more extreme or weaker versions.
    Will perturb multiple adjectives in the selected step.
    Example: 'good'->'excellent', 'bad'->'terrible', 'large'->'huge'.
    replace_prob controls how many adjectives get perturbed.
    """
    def match_case(replacement: str, original: str) -> str:
        """Match replacement's case style to original token."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        else:
            return replacement.lower()

    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)
    changed = False

    for i, (word, pos) in enumerate(tags):
        if pos.startswith('JJ') and random.random() < replace_prob:
            synsets = wn.synsets(word, pos=wn.ADJ)
            if not synsets:
                continue
            
            # collect synonyms/similar adjectives
            all_adjs = set()
            for syn in synsets:
                for lemma in syn.lemmas():
                    all_adjs.add(lemma.name().replace('_', ' '))
                for sim in syn.similar_tos():
                    for lemma in sim.lemmas():
                        all_adjs.add(lemma.name().replace('_', ' '))
            
            # remove original
            all_adjs.discard(word.lower())
            all_adjs.discard(word)
            if all_adjs:
                replacement = random.choice(list(all_adjs))
                words[i] = match_case(replacement, word)
                changed = True

    if changed:
        steps[idx] = ' '.join(words)
        return True

    return False


def perturb_meronym_confusion(steps):
    """
    Replace entities with their parts (meronyms) or wholes (holonyms).
    Example: "car" -> "wheel" (part), "wheel" -> "car" (whole)
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)

    for i, (word, pos) in enumerate(tags):
        if pos.startswith('NN'):
            synsets = wn.synsets(word, pos=wn.NOUN)
            if not synsets:
                continue
            
            # Try both part-of and has-parts relationships
            related_terms = set()
            for syn in synsets:
                # Parts of this thing (meronyms)
                for part in syn.part_meronyms():
                    for lemma in part.lemmas():
                        related_terms.add(lemma.name().replace('_', ' '))
                
                # Things this is part of (holonyms)
                for whole in syn.member_holonyms() + syn.part_holonyms():
                    for lemma in whole.lemmas():
                        related_terms.add(lemma.name().replace('_', ' '))
            
            if related_terms:
                replacement = random.choice(list(related_terms))
                words[i] = replacement
                steps[idx] = ' '.join(words)
                return True
    return False

def perturb_antonym_insertion(steps, replace_prob=0.8):
    """
    Pick verbs/adjectives with known antonyms and swap them to flip polarity.
    Example:
        'This approach increases accuracy.' ->
        'This approach decreases accuracy.'
    replace_prob controls how many eligible words get swapped.
    """
    def match_case(replacement: str, original: str) -> str:
        """Match replacement's case style to original token."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        else:
            return replacement.lower()
    
    def get_antonyms(word, pos_tag):
        """Return a list of antonyms for a given word and POS."""
        antonyms = set()
        wn_pos = None
        if pos_tag.startswith('JJ'):
            wn_pos = wn.ADJ
        elif pos_tag.startswith('VB'):
            wn_pos = wn.VERB
        elif pos_tag.startswith('RB'):
            wn_pos = wn.ADV
        else:
            return []

        for syn in wn.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                for ant in lemma.antonyms():
                    antonyms.add(ant.name().replace('_', ' '))
        return list(antonyms)

    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)
    changed = False

    for i, (word, pos) in enumerate(tags):
        ants = get_antonyms(word, pos)
        if ants and random.random() < replace_prob:
            replacement = random.choice(ants)
            words[i] = match_case(replacement, word)
            changed = True

    if changed:
        steps[idx] = ' '.join(words)
        return True

    return False

# a small pool of unrelated domain nouns you can extend freely
UNRELATED_NOUNS = [
    "zwitterion", "bandersnatch", "shoggoth", "tesseract", "kraken",
    "void-whale", "singularity", "axolotl", "ouroboros", "necronomicon",
    "doppelgänger", "hyper-croissant", "chupacabra", "zeitgeist", "mobius-strip"
]
def perturb_key_concept_swap(steps):
    """
    Replace one topical noun with an unrelated domain noun from UNRELATED_NOUNS.
    Example:
        'The neuron fires in response to stimuli.' ->
        'The volcano fires in response to stimuli.'
    """
    def match_case(replacement: str, original: str) -> str:
        """Match replacement's case style to original token."""
        if original.isupper():
            return replacement.upper()
        elif original[0].isupper():
            return replacement.capitalize()
        else:
            return replacement.lower()
    
    if not steps:
        return False

    idx = random.randrange(len(steps))
    words = steps[idx].split()
    tags = nltk.pos_tag(words)

    # collect indices of nouns
    noun_indices = [i for i, (_, pos) in enumerate(tags) if pos.startswith('NN')]
    if not noun_indices:
        return False

    # choose one noun to swap
    i = random.choice(noun_indices)
    word = words[i]

    replacement = random.choice(UNRELATED_NOUNS)
    words[i] = match_case(replacement, word)

    new_step = " ".join(words)
    if new_step != steps[idx]:
        steps[idx] = new_step
        return True

    return False

def perturb_modal_logic_confusion(steps):
    """
    Introduces modal logic confusion by manipulating modal verbs and uncertainty expressions
    to create steps that are logically compatible but epistemically problematic.
    """
    if not steps:
        return False

    # Epistemic modal transformations with different strength levels
    MODAL_TRANSFORMATIONS = [
        # Strong certainty to weak possibility (introduces doubt)
        (r"\bmust\b", "could", 0.9),
        (r"\bwill\b", "might", 0.8),
        (r"\bshall\b", "may", 0.7),
        (r"\bcertainly\b", "possibly", 0.9),
        (r"\bdefinitely\b", "perhaps", 0.8),
        (r"\bundoubtedly\b", "arguably", 0.7),
        (r"\bnecessarily\b", "conceivably", 0.6),
        
        # Weak possibility to strong certainty (over-claims)
        (r"\bcould\b", "must", 0.9),
        (r"\bmight\b", "will", 0.8),
        (r"\bmay\b", "shall", 0.7),
        (r"\bpossibly\b", "certainly", 0.9),
        (r"\bperhaps\b", "definitely", 0.8),
        (r"\barguably\b", "undoubtedly", 0.7),
        
        # Adding epistemic qualifiers to certain statements
        (r"\bis\b", "could be", 0.6),
        (r"\bare\b", "might be", 0.6),
        (r"\bwas\b", "may have been", 0.6),
        (r"\bwere\b", "could have been", 0.6),
        
        # Removing epistemic qualifiers
        (r"\bcould be\b", "is", 0.7),
        (r"\bmight be\b", "are", 0.7),
        (r"\bmay have been\b", "was", 0.7),
    ]

    # Epistemically problematic follow-up templates
    PROBLEMATIC_FOLLOWUPS = [
        "However, this could potentially be false under certain circumstances.",
        "Nevertheless, alternative interpretations might suggest otherwise.",
        "Though it should be noted that exceptions may exist.",
        "However, one could argue the opposite viewpoint.",
        "This conclusion might not hold in all possible scenarios.",
        "Though some experts might dispute this interpretation.",
        "However, counterevidence could potentially emerge.",
        "This might not be universally applicable across all contexts.",
        "Though alternative explanations could be considered.",
        "However, this perspective may be subject to revision.",
    ]

    # Strategy 1: Direct modal transformation within a step
    for i, step in enumerate(steps):
        original_step = step
        for pattern, replacement, prob in MODAL_TRANSFORMATIONS:
            if random.random() < prob and re.search(pattern, step, re.IGNORECASE):
                new_step = re.sub(pattern, replacement, step, flags=re.IGNORECASE)
                if new_step != step:
                    steps[i] = new_step
                    return True

    # Strategy 2: Add epistemically problematic follow-up step
    if len(steps) > 1:
        idx = random.randrange(len(steps) - 1)
        original_step = steps[idx]
        
        # Check if the step has some certainty or assertion
        if (re.search(r"\b(must|will|shall|is|are|was|were|certainly|definitely)\b", 
                     original_step, re.IGNORECASE) and
            not any(followup in steps[idx+1] for followup in PROBLEMATIC_FOLLOWUPS)):
            
            followup = random.choice(PROBLEMATIC_FOLLOWUPS)
            steps.insert(idx + 1, followup)
            return True

    # Strategy 3: Create modal contradiction within same step
    for i, step in enumerate(steps):
        if random.random() < 0.4:  # 40% chance to try this strategy
            # Find modal verbs in the step
            modals = re.findall(r"\b(must|will|shall|could|might|may|should|would)\b", 
                               step, re.IGNORECASE)
            
            if modals:
                # Add contradictory modal phrase
                contradictory_phrases = [
                    "although some uncertainty remains",
                    "despite potential counterarguments",
                    "while acknowledging possible exceptions",
                    "though alternative views exist",
                    "despite some contradictory evidence",
                    "while recognizing potential limitations",
                ]
                
                # Insert contradictory phrase
                words = step.split()
                if len(words) > 3:
                    insert_pos = random.randint(1, len(words) - 1)
                    contradictory_phrase = random.choice(contradictory_phrases)
                    
                    # Add comma for natural flow
                    if insert_pos > 0 and not words[insert_pos-1].endswith(','):
                        contradictory_phrase = ', ' + contradictory_phrase
                    
                    new_words = (words[:insert_pos] + 
                                [contradictory_phrase] + 
                                words[insert_pos:])
                    new_step = ' '.join(new_words)
                    
                    if new_step != step:
                        steps[i] = new_step
                        return True

    # Strategy 4: Replace factual statements with modal equivalents
    for i, step in enumerate(steps):
        if random.random() < 0.3:
            # Replace "X is Y" with "X appears to be Y" or similar
            replacements = [
                (r"\b(is|are)\b", "appears to be"),
                (r"\b(was|were)\b", "seemed to be"),
                (r"\b(proves|shows)\b", "suggests"),
                (r"\b(always|never)\b", "often"),
                (r"\b(all|every)\b", "many"),
            ]
            
            for pattern, replacement in replacements:
                if re.search(pattern, step, re.IGNORECASE):
                    new_step = re.sub(pattern, replacement, step, flags=re.IGNORECASE)
                    if new_step != step:
                        steps[i] = new_step
                        return True

    return False

# A pool of narrative/emotional appeal fragments you can expand
GENRE_MIX_FRAGMENTS = [
    "One can imagine how concerning this must be.",
    "This feels especially significant for many people.",
    "It’s easy to picture how troubling this could be.",
    "This is quite alarming from a human perspective.",
    "One might empathize with those affected.",
    "This scenario sounds almost like a story itself.",
    "It’s heartbreaking to think about the consequences.",
    "This raises deep emotional concerns for observers.",
    "One might worry about the human side of this.",
    "This situation evokes a sense of urgency."
]
def perturb_reasoning_genre_mixing(steps):
    """
    Combines formal reasoning steps with narrative/emotional appeals,
    disrupting the expected reasoning structure while maintaining topic coherence.
    Example:
        'The data shows X.' ->
        'The data shows X. One can imagine how concerning this must be.'
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    fragment = random.choice(GENRE_MIX_FRAGMENTS)

    # Only add if fragment isn't already in the step
    if fragment not in steps[idx]:
        # Randomly choose whether to append or prepend
        if random.choice([True, False]):
            steps[idx] = steps[idx] + " " + fragment
        else:
            steps[idx] = fragment + " " + steps[idx]
        return True

    return False

FINAL_VERDICTS = ["SUPPORTED", "REFUTED", "PARTIALLY SUPPORTED", "PARTIALLY REFUTED", "UNVERIFIED"]
def perturb_final_verdict_insertion(steps):
    """
    Insert a 'therefore, the Final Verdict is ___' phrase at the end of a randomly selected reasoning step.
    Example:
        'The data shows X' ->
        'The data shows X therefore, the Final Verdict is REFUTED'
    """
    if not steps:
        return False
    
    # Exclude the last step, which is usually the 'Final Verdict' line itself.
    # This prevents the final verdict from being duplicated or becoming nonsensical.
    if len(steps) > 1:
        idx = random.randrange(len(steps)-1)
    else:
        # If there's only one step, it's the only valid choice.
        idx = 0

    verdict = random.choice(FINAL_VERDICTS)
    phrase = f"therefore, the Final Verdict is {verdict}"

    # Append the phrase to the end of the step's string.
    new_step = f"{steps[idx]} {phrase}"

    # Update the step if the content has changed to prevent infinite loops in a test environment.
    if new_step != steps[idx]:
        steps[idx] = new_step
        return True
    return False

# you can extend this dictionary with any domain terms you like
DEFINITIONS = {
    "claim": "A claim is an assertion or statement that something is true, often requiring evidence or support.",
    "evidence": "Evidence is information or data presented to support or refute a claim or conclusion.",
    "statement": "A statement is a clear expression of an idea, opinion, or fact.",
    "support": "Support is material or reasoning offered to strengthen a claim or argument.",
    "plan": "A plan is a proposed set of actions or steps designed to achieve a goal or implement a policy.",
    "law": "A law is a system of rules created and enforced by a government or authority.",
    "group": "A group is a collection of people, organizations, or entities considered together.",
    "post": "A post is a message or piece of content published on social media or online platforms.",
    "government": "Government is the organization through which political authority is exercised and public policy is implemented.",
    "people": "People are individual human beings collectively or in general.",
    "time": "Time is a measure of when events occur and the duration between them.",
    "number": "A number is a mathematical value used to represent quantity, order, or measurement.",
    "part": "A part is a portion or segment of a whole.",
    "way": "A way is a manner, method, or path of doing something.",
    "work": "Work is an activity or function performed, often for employment or purpose.",
    "case": "A case is an instance, example, or legal action considered individually.",
    "year": "A year is a unit of time equal to twelve months or about 365 days.",
    "state": "A state is either a political division of a country or a particular condition or situation.",
    "job": "A job is a specific task or a position of employment.",
    "money": "Money is a medium of exchange used to buy goods and services.",
    "issue": "An issue is a topic, problem, or question under discussion or debate.",
    "policy": "A policy is a formal course of action adopted by an organization or government.",
    "vote": "A vote is a formal expression of choice or opinion in an election or decision.",
    "right": "A right is a legal, social, or ethical entitlement to have or do something.",
    "health": "Health is the state of physical, mental, and social well-being.",
    "care": "Care is the provision of attention or services to maintain or improve someone’s well-being.",
    "cost": "A cost is the amount of money or resources required to obtain or produce something.",
    "member": "A member is an individual who belongs to a group, organization, or body.",
    "office": "An office is a position of authority or a place where professional duties are carried out.",
    "bill": "A bill is a proposed law submitted for debate and approval by a legislature."
}
def get_wordnet_definition(word):
    """Fallback: return a simple definition from WordNet if available."""
    synsets = wn.synsets(word, pos=wn.NOUN)
    if not synsets:
        return None
    gloss = synsets[0].definition()
    # Capitalise word to look like a dictionary entry
    return f"{word.capitalize()} is {gloss}."
def perturb_definitional_redundancy(steps, insert_prob=0.7):
    """
    Adds dictionary-style definitions of terms already used, bloating the reasoning chain.
    Uses custom DEFINITIONS first, then WordNet as fallback.
    Only perturbs one randomly selected reasoning step at a time.
    """
    if not steps:
        return False

    # pick one step only
    idx = random.randrange(len(steps)-1)
    text = steps[idx]
    words = text.split()
    tags = nltk.pos_tag(words)
    added_any = False
 
    # collect definitions for this one step only
    definitions_to_add = []

    for (word, pos) in tags:
        if pos.startswith('NN') and random.random() < insert_prob:
            # case-insensitive check for predefined definition
            definition = None
            for term, defin in DEFINITIONS.items():
                if re.search(rf"\b{term}\b", word, flags=re.I):
                    definition = defin
                    break
            if not definition:
                # fallback to WordNet gloss
                definition = get_wordnet_definition(word.lower())

            if definition and definition not in steps:
                definitions_to_add.append(definition)
                added_any = True
                # break after first matching noun to ensure only one step perturbed
                break

    # insert collected definitions after the current step
    for defin in reversed(definitions_to_add):
        steps.insert(idx + 1, defin)

    return added_any

def simple_paraphrase(text):
    """
    Very lightweight paraphrasing:
    - Replace numbers with words or vice versa
    - Replace some common terms with trivial synonyms
    - Prepend/append tautological phrases
    """
    # number swap
    nums = re.findall(r"\d+", text)
    if nums:
        n = nums[0]
        try:
            num_int = int(n)
            # replace number with words (e.g. 10000 -> 'ten thousand')
            from num2words import num2words
            word_num = num2words(num_int)
            text = text.replace(n, word_num, 1)
        except Exception:
            pass

    # trivial synonyms map
    trivial_map = {
        r"\bpeople\b": "individuals",
        r"\bperson\b": "individual",
        r"\bpopulation\b": "group of people",
        r"\bthere are\b": "it is true that there are",
        r"\bthis\b": "this fact"
    }
    for pat, repl in trivial_map.items():
        text = re.sub(pat, repl, text, flags=re.I)

    # tautology prepend/append options
    tautologies = [
        "This implies that ",
        "It follows that ",
        "Therefore, ",
        "In other words, "
    ]
    if random.random() < 0.5:
        text = random.choice(tautologies) + text
    else:
        text = text + " (which is the same thing)."

    return text

def perturb_entailment_manipulation(steps):
    """
    Inserts a trivial paraphrase of a random step immediately after it,
    creating high entailment with no extra reasoning.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    original = steps[idx]
    paraphrased = simple_paraphrase(original)

    # only insert if the paraphrase is different and not already in steps
    if paraphrased != original and paraphrased not in steps:
        steps.insert(idx + 1, paraphrased)
        return True

    return False

def trivial_rephrase(text):
    """
    Make a cheap trivial paraphrase of a sentence:
    - Reorder some phrases
    - Reuse numbers / nouns in a slightly different wording
    - Add tautology or redundant marker
    """
    # reorder small phrases if possible
    parts = re.split(r'[,;:]|\band\b|\bbut\b', text)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        random.shuffle(parts)
        text = ', '.join(parts)

    # trivial synonyms map (lightweight)
    synonyms = {
        r"\btemperature\b": "temp",
        r"\bpeople\b": "individuals",
        r"\bchange\b": "variation",
        r"\boccurred\b": "took place",
        r"\bincreased\b": "went up",
        r"\breduced\b": "went down",
        r"\bshowed\b": "indicated",
    }
    for pat, repl in synonyms.items():
        text = re.sub(pat, repl, text, flags=re.I)

    # add tautological phrase
    tautologies = [
        "This describes the same thing.",
        "This restates the earlier point.",
        "This essentially repeats it.",
        "This refers to the same fact."
    ]
    if random.random() < 0.5:
        text = text + " " + random.choice(tautologies)
    else:
        text = random.choice(tautologies) + " " + text

    return text
def perturb_embedding_space_manipulation(steps):
    """
    Insert a semantically similar but trivial step after a random step,
    creating high embedding overlap with minimal new info.
    """
    if not steps:
        return False

    idx = random.randrange(len(steps))
    original = steps[idx]
    paraphrased = trivial_rephrase(original)

    if paraphrased != original and paraphrased not in steps:
        steps.insert(idx + 1, paraphrased)
        return True

    return False

PENULTIMATE_AMBIGUITY_STMTS = [
    "However, it is worth noting that the primary data source has been criticized for its small sample size.",
    "Nevertheless, some analysts question the reliability of this evidence.",
    "However, alternative interpretations of the findings exist.",
    "It should be considered that these results may be influenced by confounding factors.",
    "Still, other experts have proposed different explanations for these observations.",
    "Yet it remains unclear whether the data fully supports the claim.",
    "Nonetheless, doubts about the methodology could undermine this conclusion.",
    "However, this conclusion may rest on incomplete information.",
    "It is possible that biases in the data may affect the result.",
    "Another plausible explanation could contradict this inference."
]
def perturb_penultimate_ambiguity(steps):
    """
    Insert a caveat/doubt/alternative explanation right before the final step,
    undermining the belief accumulation process.
    """
    if not steps or len(steps) < 2:
        return False  # need at least 2 steps to have a "penultimate" position

    insert_idx = len(steps) - 1  # just before the final step
    stmt = random.choice(PENULTIMATE_AMBIGUITY_STMTS)

    if stmt not in steps:
        steps.insert(insert_idx, stmt)
        return True

    return False