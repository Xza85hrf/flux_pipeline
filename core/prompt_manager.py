"""Prompt management and optimization module for the FluxPipeline project.

This module provides sophisticated prompt handling capabilities for AI image generation,
including:
- Token management and optimization
- Semantic grouping of prompt components
- Style and technical keyword management
- Negative prompt processing
- Token prioritization based on semantic importance

The module ensures prompts are optimized for the model while maintaining semantic
integrity and adhering to token limits.

Example:
    Basic usage:
    ```python
    from core.prompt_manager import PromptManager
    
    # Initialize with max token length
    manager = PromptManager(max_tokens=77)
    
    # Process a positive prompt
    prompt = "A modern minimalist logo design with clean lines and bold typography"
    optimized_prompt = manager.process_prompt(prompt)
    
    # Process a negative prompt
    neg_prompt = "blurry, low quality, text, watermark"
    optimized_neg = manager.process_negative_prompt(neg_prompt)
    ```

Note:
    The module automatically handles token prioritization and ensures important
    semantic elements are preserved even when prompts need to be truncated.
"""

import re
from typing import Dict, List, Set
from config.logging_config import logger


class PromptManager:
    """Advanced prompt management system with semantic token handling.
    
    This class provides comprehensive prompt processing capabilities including:
    - Token count management
    - Semantic grouping and prioritization
    - Style and technical keyword optimization
    - Negative prompt handling
    
    Attributes:
        max_tokens (int): Maximum allowed tokens in a prompt
        token_weights (Dict[str, float]): Weight distribution for token categories
        technical_keywords (Set[str]): Technical quality-related keywords
        style_keywords (Set[str]): Style and aesthetic-related keywords
        negative_keywords (Dict[str, List[str]]): Categorized negative prompt keywords
    
    Example:
        ```python
        manager = PromptManager(max_tokens=77)
        
        # Process a complex prompt
        prompt = "Create a high quality modern logo design with bold typography, 
                 minimalist style, sharp details, and professional lighting"
        result = manager.process_prompt(prompt)
        print(f"Optimized prompt: {result}")
        ```
    """

    def __init__(self, max_tokens: int = 77):
        """Initialize the prompt manager with token constraints.
        
        Args:
            max_tokens (int, optional): Maximum number of tokens allowed in a prompt.
                Defaults to 77 (standard for many image generation models).
        """
        self.max_tokens = max_tokens
        # Weight distribution for different prompt components
        self.token_weights = {
            "subject": 0.5,    # Main subject gets highest priority
            "style": 0.25,     # Style descriptors get second priority
            "technical": 0.15,  # Technical quality terms get third priority
            "details": 0.1,    # Additional details get lowest priority
        }

        # Technical quality-related keywords
        self.technical_keywords: Set[str] = {
            "high quality",
            "sharp",
            "detailed",
            "professional",
            "resolution",
            "lighting",
            "4k",
            "8k",
            "hdr",
            "highly detailed",
            "masterpiece",
            "clear",
            "focused",
            "crisp",
            "refined",
        }

        # Style and aesthetic-related keywords
        self.style_keywords: Set[str] = {
            "modern",
            "minimalistic",
            "abstract",
            "contemporary",
            "clean",
            "realistic",
            "natural",
            "dramatic",
            "artistic",
            "elegant",
            "sophisticated",
            "simple",
            "bold",
            "subtle",
            "traditional",
        }

        # Categorized negative prompt keywords
        self.negative_keywords = {
            "style": [
                "cartoon",
                "anime",
                "sketch",
                "drawing",
                "painterly",
                "artificial",
            ],
            "quality": [
                "blurry",
                "low quality",
                "poor",
                "noisy",
                "grainy",
                "pixelated",
            ],
            "unwanted": [
                "text",
                "watermark",
                "signature",
                "artifacts",
                "distorted",
                "deformed",
            ],
        }

    def _get_token_groups(self, text: str) -> Dict[str, List[str]]:
        """Split input text into semantic token groups.
        
        This method categorizes prompt components into semantic groups for better
        token management and prioritization.
        
        Args:
            text (str): Input prompt text to be categorized
            
        Returns:
            Dict[str, List[str]]: Categorized prompt components with keys:
                - subject: Main subject or focus
                - style: Style and aesthetic descriptors
                - technical: Technical quality terms
                - details: Additional specifications
                - context: Contextual information
        
        Example:
            ```python
            text = "Modern minimalist logo with clean lines and high quality details"
            groups = manager._get_token_groups(text)
            # Result:
            # {
            #     'subject': ['Modern minimalist logo'],
            #     'style': ['clean lines'],
            #     'technical': ['high quality'],
            #     'details': ['details'],
            #     'context': []
            # }
            ```
        """
        # Normalize input text
        text = text.strip().replace("\n", " ")
        text = re.sub(r"\s+", " ", text)
        # Split into phrases at commas and periods (preserving decimals)
        phrases = [p.strip() for p in re.split(r"[,.](?!\d)", text) if p.strip()]

        # Initialize semantic groups
        groups = {
            "subject": [],
            "style": [],
            "technical": [],
            "details": [],
            "context": [],
        }

        # Categorize each phrase
        for phrase in phrases:
            words = set(phrase.lower().split())
            phrase_lower = phrase.lower()

            # Check for technical quality terms
            if any(keyword in phrase_lower for keyword in self.technical_keywords):
                groups["technical"].append(phrase)
            # Check for style descriptors
            elif any(keyword in phrase_lower for keyword in self.style_keywords):
                groups["style"].append(phrase)
            # Check for color specifications
            elif (
                "color" in phrase_lower
                or "#" in phrase
                or any(c in phrase_lower for c in ["rgb", "hsl", "cmyk"])
            ):
                groups["style"].append(phrase)
            # Check for main subject indicators
            elif len(groups["subject"]) == 0 and any(
                k in phrase_lower
                for k in ["logo", "design", "image", "picture", "photo"]
            ):
                groups["subject"].append(phrase)
            # Check for typography/text details
            elif any(
                k in phrase_lower for k in ["font", "text", "typography", "letter"]
            ):
                groups["details"].append(phrase)
            # Default categorization
            else:
                if not groups["subject"]:
                    groups["subject"].append(phrase)
                else:
                    groups["context"].append(phrase)

        return groups

    def _prioritize_tokens(
        self, token_groups: Dict[str, List[str]], target_length: int
    ) -> str:
        """Prioritize and combine tokens to fit within target length.
        
        This method implements intelligent token selection and combination
        based on semantic importance and length constraints.
        
        Args:
            token_groups (Dict[str, List[str]]): Categorized token groups
            target_length (int): Maximum allowed length for combined tokens
            
        Returns:
            str: Optimized and combined token string
        
        Example:
            ```python
            groups = {
                'subject': ['Modern logo'],
                'style': ['minimalist', 'clean'],
                'technical': ['high quality'],
                'details': ['bold typography']
            }
            result = manager._prioritize_tokens(groups, 50)
            ```
        """
        # Check if total length is within limit
        total_tokens = sum(len(" ".join(group)) for group in token_groups.values())
        if total_tokens <= target_length:
            return ", ".join(
                [item for group in token_groups.values() for item in group]
            )

        # Allocate tokens based on weights
        allocated_tokens = {
            category: max(10, int(target_length * weight))
            for category, weight in self.token_weights.items()
        }

        # Adjust allocation if over target
        total_allocated = sum(allocated_tokens.values())
        if total_allocated > target_length:
            scale_factor = target_length / total_allocated
            allocated_tokens = {
                k: max(5, int(v * scale_factor)) for k, v in allocated_tokens.items()
            }

        final_phrases = []
        # Process each category
        for category, phrases in token_groups.items():
            if not phrases:
                continue

            category_text = ", ".join(phrases)
            if len(category_text) > allocated_tokens[category]:
                words = category_text.split()
                current_length = 0
                selected_words = []

                # Select words based on importance
                for i, word in enumerate(words):
                    is_important = (
                        i == 0  # First word
                        or i == len(words) - 1  # Last word
                        or any(  # Technical or style keyword
                            k in word.lower()
                            for k in self.technical_keywords | self.style_keywords
                        )
                    )

                    # Add word if within length limit
                    if current_length + len(word) + 1 <= allocated_tokens[category]:
                        if (
                            is_important
                            or len(selected_words) < allocated_tokens[category] // 2
                        ):
                            selected_words.append(word)
                            current_length += len(word) + 1
                    # Replace less important word if needed
                    elif is_important:
                        while (
                            selected_words
                            and current_length + len(word) + 1
                            > allocated_tokens[category]
                        ):
                            removed_word = selected_words.pop(len(selected_words) // 2)
                            current_length -= len(removed_word) + 1
                        selected_words.append(word)
                        current_length += len(word) + 1

                category_text = " ".join(selected_words)

            if category_text:
                final_phrases.append(category_text)

        return ", ".join(phrase for phrase in final_phrases if phrase)

    def process_prompt(self, prompt: str) -> str:
        """Process and optimize prompt for generation.
        
        This method handles the complete prompt optimization process:
        1. Normalizes input
        2. Ensures quality terms are present
        3. Manages token count
        4. Preserves semantic integrity
        
        Args:
            prompt (str): Input prompt to process
            
        Returns:
            str: Optimized prompt string
        
        Example:
            ```python
            prompt = "Create a modern minimalist logo with clean lines"
            result = manager.process_prompt(prompt)
            print(f"Optimized: {result}")
            # Output: "high quality, detailed, modern minimalist logo, clean lines"
            ```
        """
        if not prompt:
            return ""

        # Normalize prompt
        prompt = prompt.strip()
        prompt = " ".join(prompt.split())

        # Add quality terms if missing
        quality_terms = ["high quality", "detailed", "high resolution"]
        if not any(term in prompt.lower() for term in quality_terms):
            prompt = f"high quality, detailed, {prompt}"

        # Check token count
        prompt_tokens = prompt.split()
        if len(prompt_tokens) <= self.max_tokens:
            return prompt

        # Distribute tokens by category
        main_subject = prompt_tokens[
            : int(self.max_tokens * self.token_weights["subject"])
        ]
        style_terms = [
            t
            for t in prompt_tokens
            if any(s in t.lower() for s in ["style", "design", "look", "aesthetic"])
        ][: int(self.max_tokens * self.token_weights["style"])]
        technical_terms = [
            t
            for t in prompt_tokens
            if any(t in t.lower() for t in ["quality", "resolution", "detailed"])
        ][: int(self.max_tokens * self.token_weights["technical"])]

        # Combine tokens
        final_tokens = []
        final_tokens.extend(main_subject)
        final_tokens.extend(style_terms)
        final_tokens.extend(technical_terms)

        # Trim to fit token limit
        while len(" ".join(final_tokens)) > self.max_tokens:
            if len(final_tokens) > 2:
                final_tokens.pop()
            else:
                break

        return " ".join(final_tokens)

    def process_negative_prompt(self, prompt: str) -> str:
        """Process and optimize negative prompt.
        
        This method handles negative prompt optimization:
        1. Identifies predefined negative terms
        2. Preserves custom negative terms
        3. Manages token count
        
        Args:
            prompt (str): Input negative prompt
            
        Returns:
            str: Optimized negative prompt
        
        Example:
            ```python
            neg_prompt = "blurry, low quality, text, watermark, bad composition"
            result = manager.process_negative_prompt(neg_prompt)
            print(f"Optimized negative: {result}")
            ```
        """
        if not prompt:
            return ""

        # Normalize prompt
        prompt = prompt.strip()
        prompt = re.sub(r"\s+", " ", prompt)

        # Find predefined negative terms
        found_terms = []
        for category, keywords in self.negative_keywords.items():
            for keyword in keywords:
                if keyword.lower() in prompt.lower():
                    found_terms.append(keyword)

        # Extract custom terms
        custom_terms = [
            term.strip()
            for term in prompt.split(",")
            if not any(
                keyword in term.lower()
                for keywords in self.negative_keywords.values()
                for keyword in keywords
            )
        ]

        # Combine terms
        all_terms = found_terms + custom_terms
        if not all_terms:
            return prompt

        # Check token limit
        combined_prompt = ", ".join(all_terms)
        if len(combined_prompt) <= self.max_tokens:
            return combined_prompt

        # Prioritize terms
        important_terms = found_terms[: self.max_tokens // 2]
        remaining_space = self.max_tokens - len(", ".join(important_terms))

        # Add custom terms if space allows
        for term in custom_terms:
            if len(term) + 2 <= remaining_space:
                important_terms.append(term)
                remaining_space -= len(term) + 2
            else:
                break

        return ", ".join(important_terms)
