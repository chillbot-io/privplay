#!/usr/bin/env python3
"""
Hard Negatives Generator for PII-BERT Training

Generates samples that LOOK like PII but AREN'T.
This teaches the model boundaries - when NOT to fire.

Categories:
1. Name-like common words (April, Bill, Chase, Major)
2. Numbers in non-PII contexts (years, room numbers, product codes)
3. PII-like patterns that aren't sensitive (public numbers, examples)
4. Instructions/descriptions about PII (no actual PII)
5. Partial patterns and edge cases

Usage:
    python3 create_hard_negatives.py --count 10000 --output hard_negatives.json
"""

import json
import random
from typing import List, Dict, Tuple
from collections import Counter

# =============================================================================
# CATEGORY 1: Name-like words that AREN'T names
# =============================================================================

# Months and days that are also names
NAME_LIKE_MONTHS = ["April", "May", "June", "August"]
NAME_LIKE_DAYS = ["Tuesday", "Friday"]  # Less common but exist

# Common words that are also names
NAME_LIKE_VERBS = [
    "Bill", "Will", "Mark", "Rob", "Sue", "Pat", "Drew", "Grant", "Skip",
    "Chase", "Wade", "Pierce", "Lance", "Marshall", "Park", "Foster",
]

NAME_LIKE_NOUNS = [
    "Rose", "Iris", "Lily", "Daisy", "Holly", "Ivy", "Olive", "Violet",
    "Ruby", "Pearl", "Crystal", "Amber", "Jade", "Jasper", "Clay",
    "Brook", "River", "Lake", "Glen", "Dale", "Forest", "Stone", "Rock",
    "Major", "Bishop", "King", "Duke", "Prince", "Earl", "Baron",
    "North", "South", "West", "East", "Forward", "Chase", "Banks",
]

NAME_LIKE_ADJECTIVES = [
    "Rich", "Noble", "Hardy", "Young", "Long", "Short", "Strong",
    "Bright", "Fair", "True", "Gay", "Hardy", "Stern", "Sharp",
]

# Company names that look like person names
COMPANY_NAMES = [
    "Johnson & Johnson", "Wells Fargo", "Chase Bank", "Morgan Stanley",
    "Goldman Sachs", "Charles Schwab", "Edward Jones", "Raymond James",
    "Jack Daniels", "Ben & Jerry", "Mary Kay", "Martha Stewart",
]

# Contexts where these appear as NOT names
NAME_LIKE_CONTEXTS = [
    # Months
    "The meeting is scheduled for {word} 15th.",
    "We expect delivery in {word}.",
    "{word} showers bring May flowers.",
    "The deadline is {word} 30.",
    "File your taxes before {word}.",
    
    # Verbs
    "{word} the client immediately.",
    "Please {word} this to the team.",
    "Can you {word} the report?",
    "I'll {word} it down for you.",
    "{word} this email for reference.",
    "Don't forget to {word} the changes.",
    
    # Nouns (objects/places)
    "The {word} garden is beautiful.",
    "Pick up the {word} on your way.",
    "The {word} is blooming nicely.",
    "Look at that {word} over there.",
    "The {word} bridge needs repair.",
    "Down by the {word} valley.",
    
    # Adjectives
    "The {word} man helped us.",
    "She has a {word} future ahead.",
    "That's a {word} decision.",
    
    # Companies/Products
    "I bank with {word}.",
    "My account is at {word}.",
    "Transfer funds to {word}.",
    "The {word} representative called.",
]


# =============================================================================
# CATEGORY 2: Numbers that AREN'T PII
# =============================================================================

NON_PII_NUMBER_CONTEXTS = [
    # Years
    "The company was founded in {num}.",
    "Back in {num}, things were different.",
    "Copyright {num} All Rights Reserved.",
    "Established {num}.",
    "Since {num}.",
    "The year was {num}.",
    "Published in {num}.",
    "Class of {num}.",
    
    # Room/Building numbers
    "Meet me in Room {num}.",
    "Conference Room {num} is available.",
    "Building {num}, Floor 3.",
    "Office {num} is down the hall.",
    "Please report to Gate {num}.",
    "Your seat is {num}A.",
    "Locker {num} is yours.",
    
    # Product/Order numbers
    "Order #{num}.",
    "Product ID: {num}",
    "Invoice {num}.",
    "Receipt #{num}",
    "Confirmation number: {num}",
    "Ticket #{num}",
    "Case #{num}",
    "Reference: {num}",
    "SKU: {num}",
    "Part number {num}.",
    "Model {num}.",
    "Serial: {num}",  # NOT SSN despite format
    "Item #{num}",
    
    # Measurements/Statistics
    "The temperature was {num} degrees.",
    "About {num} people attended.",
    "Sales reached {num} units.",
    "Distance: {num} miles.",
    "Weight: {num} pounds.",
    "Height: {num} feet.",
    "The score was {num} to 42.",
    
    # Prices (not card numbers)
    "Total: ${num}",
    "Price: {num} USD",
    "Cost: ${num}.99",
    "Budget: {num}",
    
    # Codes (not SSN)
    "Error code {num}.",
    "Status code: {num}",
    "Exit code {num}.",
    "ZIP code lookup for {num}.",  # Mentioned but as lookup, not actual address
    
    # Public/Emergency numbers
    "Call 911 for emergencies.",
    "Dial 411 for information.",
    "Call 311 for city services.",
    "The operator is at 0.",
]


# =============================================================================
# CATEGORY 3: PII-like patterns that AREN'T sensitive
# =============================================================================

NON_SENSITIVE_PATTERNS = [
    # Example/placeholder data
    "Example SSN: XXX-XX-XXXX",
    "Format: 123-45-6789 (not real)",
    "Sample card: 4111-1111-1111-1111",
    "Use test@example.com for testing.",
    "Demo account: user@test.com",
    "Placeholder: john.doe@example.com",
    
    # Masked/redacted data
    "SSN: ***-**-1234",
    "Card ending in ****1234",
    "Phone: (XXX) XXX-1234",
    "Email: j***@gmail.com",
    
    # Public/Generic contact info
    "Contact support@company.com for help.",
    "Email info@example.org for details.",
    "General inquiries: contact@business.com",
    "Support hotline: 1-800-555-0199",
    "Customer service: 1-888-555-0123",
    "Toll-free: 1-800-EXAMPLE",
    
    # Documentation/Instructions
    "Your phone number should be 10 digits.",
    "SSN must be in XXX-XX-XXXX format.",
    "Enter your 16-digit card number.",
    "Password must be at least 8 characters.",
    "Email addresses contain an @ symbol.",
    "US phone numbers start with area code.",
    
    # Form labels without values
    "Phone Number: _____________",
    "Email Address: _____________", 
    "SSN: ___ - __ - ____",
    "Credit Card: ____ ____ ____ ____",
    "Name: _____________",
    
    # Descriptions about PII
    "Social Security numbers are 9 digits.",
    "Credit cards use Luhn validation.",
    "Phone numbers vary by country.",
    "Email addresses have a local and domain part.",
    "IP addresses have four octets.",
    
    # Fictional/Obvious fakes
    "John Doe is a placeholder name.",
    "Jane Smith is commonly used in examples.",
    "The password 'password123' is insecure.",
    "Test accounts use fake data.",
    
    # Old/Invalid formats  
    "The old extension was x1234.",
    "Previous ID: 00000000",
    "Deprecated code: 999-99-9999",
]


# =============================================================================
# CATEGORY 4: Partial/Incomplete patterns
# =============================================================================

PARTIAL_PATTERNS = [
    # Partial phone numbers
    "Extension 1234.",
    "Dial ext. 5678.",
    "Press 1 for sales.",
    "Enter the last 4 digits.",
    "Code: 0000.",
    
    # Partial SSN references
    "Last 4 of SSN: 1234",
    "SSN ending in 5678",
    "Verify last four: XXXX",
    
    # Short number sequences
    "PIN: 1234",  # Could be, but in isolation it's ambiguous
    "Code 5678",
    "Enter 0000 to reset.",
    "The answer is 42.",
    "Channel 7 news.",
    "Highway 101.",
    "Route 66.",
    "Page 123.",
    "Chapter 7.",
    "Section 8.",
    "Paragraph 3.",
    "Version 2.0.",
    "Release 3.14.",
    
    # Generic alphanumeric
    "ID: ABC123",
    "Code: XY789",
    "Ref: Z12345",
]


# =============================================================================
# CATEGORY 5: Context confusion cases
# =============================================================================

CONFUSION_CASES = [
    # Address-like but not addresses
    "Main Street is busy today.",
    "Take Oak Avenue to the highway.",
    "The Park Street bridge is closed.",
    "I live near Broadway.",
    "Turn left on First Street.",
    
    # Date-like but not dates
    "We met around 2015.",
    "The 90s were different.",
    "It happened in the early 2000s.",
    "From 9 to 5.",
    "Between 1 and 10.",
    "About 3 weeks ago.",
    
    # Email-like but not emails
    "The ratio is 3@2 for best results.",  # @ in non-email context
    "Contact us via our website.",
    "Reach out through the portal.",
    "Message us on social media.",
    
    # Password-like but instructions
    "Passwords should include numbers.",
    "Use a mix of letters and symbols.",
    "Avoid common words like 'password'.",
    "Don't use '123456' as a password.",
    
    # Account-like but not accounts
    "Your score is 98765432.",
    "The population is 12345678.",
    "Record ID: 87654321.",
    "Batch number 11223344.",
    
    # IP-like but version numbers
    "Version 1.2.3.4 is available.",
    "Update to 10.0.0.1",
    "Release 192.168.1.1 notes.",  # Looks like IP but is version
]


# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

def generate_name_like_negative() -> Dict:
    """Generate a sample with name-like word that isn't a name."""
    category = random.choice([
        ("month", NAME_LIKE_MONTHS),
        ("verb", NAME_LIKE_VERBS),
        ("noun", NAME_LIKE_NOUNS),
        ("adjective", NAME_LIKE_ADJECTIVES),
        ("company", COMPANY_NAMES),
    ])
    
    cat_type, words = category
    word = random.choice(words)
    
    # Choose appropriate context
    if cat_type == "month":
        contexts = [c for c in NAME_LIKE_CONTEXTS if "showers" in c or "deadline" in c or "delivery" in c or "scheduled" in c or "taxes" in c]
    elif cat_type == "verb":
        contexts = [c for c in NAME_LIKE_CONTEXTS if "Please" in c or "Can you" in c or "Don't" in c or "I'll" in c]
    elif cat_type == "company":
        contexts = [c for c in NAME_LIKE_CONTEXTS if "bank" in c or "account" in c or "representative" in c]
    else:
        contexts = [c for c in NAME_LIKE_CONTEXTS if "The" in c or "garden" in c or "bridge" in c or "valley" in c or "future" in c or "decision" in c]
    
    if not contexts:
        contexts = NAME_LIKE_CONTEXTS
    
    template = random.choice(contexts)
    text = template.replace("{word}", word)
    
    return {
        "text": text,
        "entities": [],  # No entities - this is NOT PII
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": f"name_like_{cat_type}",
    }


def generate_number_negative() -> Dict:
    """Generate a sample with numbers that aren't PII."""
    template = random.choice(NON_PII_NUMBER_CONTEXTS)
    
    # Generate appropriate number based on context
    if "founded" in template or "year" in template or "Copyright" in template or "Since" in template or "Class" in template:
        num = str(random.randint(1950, 2024))
    elif "Room" in template or "Gate" in template or "Building" in template or "Office" in template:
        num = str(random.randint(100, 999))
    elif "Order" in template or "Invoice" in template or "Confirmation" in template or "Reference" in template:
        num = str(random.randint(100000, 9999999))
    elif "SKU" in template or "Part" in template or "Serial" in template:
        num = f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
    elif "$" in template or "USD" in template or "Price" in template:
        num = str(random.randint(10, 9999))
    elif "code" in template.lower():
        num = str(random.randint(100, 999))
    else:
        num = str(random.randint(1, 9999))
    
    text = template.replace("{num}", num)
    
    return {
        "text": text,
        "entities": [],
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": "non_pii_number",
    }


def generate_pattern_negative() -> Dict:
    """Generate a sample with PII-like patterns that aren't sensitive."""
    text = random.choice(NON_SENSITIVE_PATTERNS)
    
    return {
        "text": text,
        "entities": [],
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": "non_sensitive_pattern",
    }


def generate_partial_negative() -> Dict:
    """Generate a sample with partial/incomplete patterns."""
    text = random.choice(PARTIAL_PATTERNS)
    
    return {
        "text": text,
        "entities": [],
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": "partial_pattern",
    }


def generate_confusion_negative() -> Dict:
    """Generate a context confusion case."""
    text = random.choice(CONFUSION_CASES)
    
    return {
        "text": text,
        "entities": [],
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": "confusion_case",
    }


def generate_mixed_negative() -> Dict:
    """Generate a longer sample mixing clean text with PII-like patterns."""
    clean_sentences = [
        "Thank you for your inquiry.",
        "We appreciate your business.",
        "Please let us know if you have questions.",
        "The team will review your request.",
        "We look forward to hearing from you.",
        "Best regards from the management team.",
        "Your feedback is important to us.",
        "We value your partnership.",
        "The project is on track for completion.",
        "Our office hours are 9 AM to 5 PM.",
        "The meeting has been rescheduled.",
        "Please find attached the documents.",
        "We have received your application.",
        "Your request has been processed.",
        "Thank you for your patience.",
    ]
    
    # Combine 2-3 clean sentences
    sentences = random.sample(clean_sentences, random.randint(2, 3))
    text = " ".join(sentences)
    
    return {
        "text": text,
        "entities": [],
        "quality_score": 0.99,
        "source": "hard_negative",
        "negative_type": "clean_text",
    }


# Generator weights
NEGATIVE_GENERATORS = [
    (generate_name_like_negative, 2.5),
    (generate_number_negative, 2.5),
    (generate_pattern_negative, 2.0),
    (generate_partial_negative, 1.5),
    (generate_confusion_negative, 1.0),
    (generate_mixed_negative, 0.5),
]


def create_hard_negatives_dataset(count: int, output_path: str):
    """Create hard negatives dataset."""
    from rich.console import Console
    from rich.progress import Progress
    
    console = Console()
    console.print(f"\n[bold cyan]═══ Generating Hard Negatives ═══[/bold cyan]\n")
    
    samples = []
    type_counts = Counter()
    
    # Build weighted generator list
    generators = []
    weights = []
    for gen, weight in NEGATIVE_GENERATORS:
        generators.append(gen)
        weights.append(weight)
    
    total_weight = sum(weights)
    probs = [w / total_weight for w in weights]
    
    with Progress() as progress:
        task = progress.add_task("Generating...", total=count)
        
        while len(samples) < count:
            # Choose generator
            gen = random.choices(generators, weights=probs, k=1)[0]
            
            try:
                sample = gen()
                samples.append(sample)
                type_counts[sample.get("negative_type", "unknown")] += 1
            except Exception as e:
                pass
            
            progress.update(task, completed=len(samples))
    
    # Shuffle
    random.shuffle(samples)
    
    # Save
    output = {
        "samples": samples,
        "metadata": {
            "total_samples": len(samples),
            "source": "hard_negatives",
            "type_distribution": dict(type_counts),
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    console.print(f"\n[bold]Hard Negative Types:[/bold]")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        console.print(f"  {t:25} {c:5}")
    
    console.print(f"\n[green]✓ Saved {len(samples)} hard negatives to {output_path}[/green]\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=10000)
    parser.add_argument("--output", type=str, default="hard_negatives.json")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    create_hard_negatives_dataset(args.count, args.output)
