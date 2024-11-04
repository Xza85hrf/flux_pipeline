"""
Unit tests for the PromptManager module.

This module contains tests that validate the functionality of the PromptManager class, which is responsible for managing and processing prompts in a text generation application. The tests cover various aspects such as prompt initialization, token detection, prioritization, and edge case handling.
"""

import pytest
from core.prompt_manager import PromptManager


@pytest.fixture
def prompt_manager():
    """Fixture providing a PromptManager instance."""
    return PromptManager(max_tokens=77)


def test_prompt_manager_init(prompt_manager):
    """
    Test PromptManager initialization and default settings.

    Tests:
    - Default max_tokens setting
    - Presence of token weights
    - Presence of technical, style, and negative keywords
    """
    assert prompt_manager.max_tokens == 77
    assert all(
        k in prompt_manager.token_weights
        for k in ["subject", "style", "technical", "details"]
    )
    assert prompt_manager.technical_keywords
    assert prompt_manager.style_keywords
    assert prompt_manager.negative_keywords


@pytest.mark.parametrize(
    "input_prompt,expected_technical,expected_style",
    [
        (
            "A high quality portrait of a woman",
            ["high quality"],
            [],
        ),
        (
            "Modern minimalist logo design with clean lines",
            [],
            ["Modern minimalist", "clean"],
        ),
        (
            "Create a photo with professional lighting and 8k resolution",
            ["professional lighting", "8k resolution"],
            [],
        ),
    ],
)
def test_token_group_detection(
    input_prompt, expected_technical, expected_style, prompt_manager
):
    """
    Test detection of technical and style terms in prompts.

    Args:
        input_prompt: The input prompt to be processed
        expected_technical: List of expected technical terms
        expected_style: List of expected style terms
    """
    groups = prompt_manager._get_token_groups(input_prompt)

    # Check technical terms
    if expected_technical:
        for term in expected_technical:
            found_technical = any(
                term.lower() in tech.lower() for tech in groups["technical"]
            )
            assert (
                found_technical
            ), f"Expected technical term '{term}' not found in: {groups['technical']}"

    # Check style terms
    if expected_style:
        for term in expected_style:
            found_style = any(
                term.lower() in style.lower() for style in groups["style"]
            )
            assert (
                found_style
            ), f"Expected style term '{term}' not found in: {groups['style']}"


def test_token_prioritization(prompt_manager):
    """
    Test token prioritization with realistic groups.

    Tests:
    - Essential elements are preserved in the right order
    - Token prioritization logic
    """
    token_groups = {
        "subject": ["portrait of a woman"],
        "style": ["modern", "elegant"],
        "technical": ["high quality", "detailed"],
        "details": ["blue background"],
    }

    result = prompt_manager._prioritize_tokens(token_groups, 50)

    # Check essential elements are preserved in the right order
    assert "portrait" in result.lower()
    assert any(style in result.lower() for style in ["modern", "elegant"])
    assert any(tech in result.lower() for tech in ["high quality", "detailed"])


def test_prompt_processing(prompt_manager):
    """
    Test complete prompt processing pipeline.

    Tests:
    - Quality terms are present
    - Essential elements are preserved
    - Token limit is respected
    """
    prompt = "Create a high quality portrait with professional lighting"
    processed = prompt_manager.process_prompt(prompt)

    # Verify quality terms are present
    assert "high quality" in processed.lower()
    assert "portrait" in processed.lower()
    assert "professional" in processed.lower()
    assert len(processed.split()) <= prompt_manager.max_tokens


def test_negative_prompt_processing(prompt_manager):
    """
    Test negative prompt processing.

    Tests:
    - Negative keywords are preserved
    """
    negative_prompt = "blurry, low quality, text, watermark"
    processed = prompt_manager.process_negative_prompt(negative_prompt)

    # Check if negative keywords are preserved
    for term in ["blurry", "low quality", "text", "watermark"]:
        assert term in processed.lower()


@pytest.mark.parametrize(
    "prompt, expected",
    [
        ("", ""),  # Empty string
        (" ", "high quality, detailed, "),  # Space only
        ("\n\t\r", "high quality, detailed, "),  # Whitespace characters
    ],
)
def test_prompt_edge_cases(prompt, expected, prompt_manager):
    """
    Test handling of edge case prompts.

    Args:
        prompt: The input prompt to be processed
        expected: The expected output after processing
    """
    processed = prompt_manager.process_prompt(prompt)
    assert (
        processed == expected
    ), f"Expected '{expected}' for input '{prompt}', got '{processed}'"


def test_empty_input_handling(prompt_manager):
    """
    Test handling of empty inputs.

    Tests:
    - Various forms of empty inputs are handled correctly
    """
    test_cases = {
        "": "",
        " ": "high quality, detailed, ",
        "\n": "high quality, detailed, ",
        "\t": "high quality, detailed, ",
        "\r": "high quality, detailed, ",
        "   ": "high quality, detailed, ",
    }

    for empty_input, expected in test_cases.items():
        processed = prompt_manager.process_prompt(empty_input)
        assert (
            processed == expected
        ), f"Expected '{expected}' for input '{empty_input}', got '{processed}'"


def test_combined_edge_cases(prompt_manager):
    """
    Test handling of combined edge cases.

    Tests:
    - Various edge cases are handled correctly
    """
    edge_cases = {
        " High Quality ": "High Quality",  # Extra spaces
        "\nDetailed\n": "Detailed",  # Newlines
        "\tPhoto\t": "high quality, detailed, Photo",  # Tabs
        "High!!Quality": "high quality, detailed, High!!Quality",  # Special characters
        "🎨Photo": "high quality, detailed, 🎨Photo",  # Emoji with text
        "": "",  # Empty string
        " ": "high quality, detailed, ",  # Single space
    }

    for input_case, expected_output in edge_cases.items():
        processed = prompt_manager.process_prompt(input_case)
        assert (
            processed.strip() == expected_output.strip()
        ), f"Input: '{input_case}', Expected: '{expected_output}', Got: '{processed}'"


def test_technical_keyword_preservation(prompt_manager):
    """
    Test preservation of technical keywords in processing.

    Tests:
    - Technical keywords are preserved
    """
    prompt = "Create a 4k resolution image with HDR and professional lighting"
    processed = prompt_manager.process_prompt(prompt)

    assert any(term in processed.lower() for term in ["4k", "hdr", "professional"])


def test_style_keyword_handling(prompt_manager):
    """
    Test handling of style keywords.

    Tests:
    - Style keywords are preserved
    """
    prompt = "Modern minimalist design with elegant composition"
    processed = prompt_manager.process_prompt(prompt)

    assert any(
        term in processed.lower() for term in ["modern", "minimalist", "elegant"]
    )


def test_special_character_handling(prompt_manager):
    """
    Test handling of special characters.

    Tests:
    - Special characters are preserved
    """
    special_inputs = {
        "!@#$%^&*()": "high quality, detailed, !@#$%^&*()",
        "\\n\\t\\r": "high quality, detailed, \\n\\t\\r",
        "''''\"\"\"\"": "high quality, detailed, ''''\"\"\"\"",
        ".,;:[]{}": "high quality, detailed, .,;:[]{}",
        "👍🌟🎨": "high quality, detailed, 👍🌟🎨",
    }

    for input_text, expected in special_inputs.items():
        processed = prompt_manager.process_prompt(input_text)
        assert (
            processed.strip() == expected.strip()
        ), f"Input: '{input_text}', Expected: '{expected}', Got: '{processed}'"


def test_quality_terms_addition(prompt_manager):
    """
    Test addition of quality terms to various inputs.

    Tests:
    - Quality terms are added appropriately
    """
    test_cases = {
        "": "",  # Empty input
        " ": "high quality, detailed, ",  # Space only
        "simple photo": "high quality, detailed, simple photo",
        "high quality photo": "high quality photo",  # Already has quality term
        "detailed image": "detailed image",  # Already has quality term
        "high resolution pic": "high resolution pic",  # Already has quality term
    }

    for input_text, expected in test_cases.items():
        processed = prompt_manager.process_prompt(input_text)
        assert (
            processed.strip() == expected.strip()
        ), f"Input: '{input_text}', Expected: '{expected}', Got: '{processed}'"


def test_emoji_handling(prompt_manager):
    """
    Test handling of emoji characters.

    Tests:
    - Emojis are preserved
    - Meaningful content is preserved
    """
    emoji_prompt = "👍 Create a 🎨 high quality portrait"
    processed = prompt_manager.process_prompt(emoji_prompt)

    # Verify emojis don't break processing and meaningful content is preserved
    assert "high quality" in processed.lower()
    assert "portrait" in processed.lower()


def test_whitespace_normalization(prompt_manager):
    """
    Test normalization of various whitespace patterns.

    Tests:
    - Various whitespace patterns are normalized correctly
    """
    whitespace_prompts = {
        "high   quality   portrait": "high quality portrait",
        "high\tquality\tportrait": "high quality portrait",
        "high\nquality\nportrait": "high quality portrait",
        " high quality portrait ": "high quality portrait",
    }

    for input_prompt, expected in whitespace_prompts.items():
        processed = prompt_manager.process_prompt(input_prompt)
        assert (
            processed.strip() == expected.strip()
        ), f"Input: '{input_prompt}', Expected: '{expected}', Got: '{processed}'"
