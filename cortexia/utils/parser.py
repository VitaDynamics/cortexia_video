import ast
from typing import List

def parse_string_like_list(data_string: str) -> List[str]:
  """
  Safely parses a string that represents a Python list of strings
  and returns a list of the extracted words.

  Args:
    data_string: A string that looks like a Python list of strings
                 (e.g., "['office', 'desk', ...]").

  Returns:
    A list of strings extracted from the input string.
    Returns an empty list if parsing fails or the string is not a list.
  """
  try:
    # Safely evaluate the string as a Python literal
    parsed_data = ast.literal_eval(data_string)

    # Check if the evaluated result is a list and contains only strings
    if isinstance(parsed_data, list) and all(isinstance(item, str) for item in parsed_data):
      return parsed_data
    else:
      print("Warning: The string did not evaluate to a list of strings.")
      return []

  except (ValueError, SyntaxError) as e:
    print(f"Error parsing the string: {e}")
    print("Please ensure the string is correctly formatted like a Python list of strings.")
    return []


def parse_comma_separated_string(data_string: str) -> List[str]:
    """
    Parses a string of comma-separated words (potentially quoted)
    and returns a list of cleaned words.

    Args:
        data_string: A string with words separated by commas.

    Returns:
        A list of cleaned strings.
    """
    # Remove potential outer brackets or excessive whitespace
    cleaned_string = data_string.strip().strip('[]')

    # Split by comma
    words = cleaned_string.split(',')

    # Clean up each word (remove quotes, leading/trailing whitespace)
    cleaned_words = [word.strip().strip("'").strip('"') for word in words]

    return cleaned_words

def generate_formatted_string_dino(word_list: List[str]) -> str:
  """
  Formats a list of words into a string like "word1. Word2. Word3.".
  The first word is lowercase, subsequent words are capitalized,
  separated by ". ", and ends with a period.

  Args:
    word_list: A list of strings (words).

  Returns:
    A formatted string or an empty string if the input list is empty.
  """
  if not word_list:
    return ""

  formatted_parts = []

  # Process the first word (lowercase)
  formatted_parts.append(word_list[0].lower())

  # Process the rest of the words (capitalize the first letter)
  for word in word_list[1:]:
    if word: # Ensure the word is not empty
      # Capitalize the first letter and keep the rest lowercase
      formatted_parts.append(word[0].upper() + word[1:].lower())
    else:
      # Handle cases with empty strings in the list if necessary
      formatted_parts.append("") # Or skip, depending on desired behavior

  # Join the parts with ". " and add a final period
  return ". ".join(formatted_parts) + "."