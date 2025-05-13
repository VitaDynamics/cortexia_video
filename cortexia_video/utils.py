def convert_to_runs(data):
  """
  Converts detection results data into a list of 'runs' dictionaries.

  Args:
    data: A list containing a dictionary with detection results
          including 'boxes' and 'text_labels'.

  Returns:
    A list of dictionaries, where each dictionary represents a 'run'
    with 'use_box', 'box', and 'output_image_path'.
  """
  runs = []
  if not data or 'boxes' not in data[0]:
    return runs

  boxes = data[0]['boxes']

  for i, box in enumerate(boxes):
    # Ensure box is a list of numbers, regardless of input type (tensor or list)
    # If input was a tensor, .tolist() would handle it. If it's already a list,
    # list comprehension or casting ensures it's a list of the correct types.
    box_list = [int(coord) for coord in box]


    run = {
        'use_box': True,
        'box': box_list,
        # Placeholder output path, can be made more dynamic if needed
        # TODO: Modify the output path to be more dynamic
        'output_image_path': f'output_visualization_box_{i}.png'
    }
    runs.append(run)

  return runs