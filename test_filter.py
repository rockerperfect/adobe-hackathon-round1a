#!/usr/bin/env python3
"""Test the specific filter logic for overview section."""

def test_overview_filter():
    text = "Overview of the Foundation Level Extension –"
    text_clean = text.strip()
    text_lower = text_clean.lower()
    
    print(f"Testing text: '{text}'")
    print(f"text_clean: '{text_clean}'")
    print(f"text_lower: '{text_lower}'")
    print(f"Ends with dash: {text_clean.endswith('–') or text_clean.endswith('-')}")
    print(f"Contains 'overview': {'overview' in text_lower}")
    print(f"Word count: {len(text_clean.split())}")
    print(f"Words: {text_clean.split()}")
    
    # Test the exact condition
    if text_clean.endswith('–') or text_clean.endswith('-'):
        # Allow if it's a substantial overview section
        condition = 'overview' in text_lower and len(text_clean.split()) >= 5
        print(f"Condition result: {condition}")
        if not condition:
            print("WOULD BE REJECTED")
            return False
        else:
            print("WOULD BE ALLOWED")
            return True
    else:
        print("DOESN'T END WITH DASH")
        return True

if __name__ == "__main__":
    test_overview_filter()
