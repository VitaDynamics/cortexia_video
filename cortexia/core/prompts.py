import functools


def with_prompt(default_prompt=None):
    """
    A decorator that modifies a class's __init__ method to accept a 'prompt' keyword argument.

    Args:
        default_prompt (str, optional): The default prompt to use if none is provided
                                        to the constructor. Defaults to None.
    """

    def decorator(cls):
        original_init = cls.__init__

        @functools.wraps(original_init)
        def new_init(self, *args, **kwargs):
            # Pop the prompt from kwargs, using the decorator's default if not present.
            prompt = kwargs.pop("prompt", default_prompt)

            # Call the original __init__ method.
            original_init(self, *args, **kwargs)

            # Set the prompt on the instance *after* the original __init__ has run.
            self.prompt = prompt

        # Replace the original __init__ with the new one.
        cls.__init__ = new_init
        return cls

    return decorator
