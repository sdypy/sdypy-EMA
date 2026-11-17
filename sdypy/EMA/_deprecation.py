"""Deprecated aliases for renamed public attributes (SEP 2).

The old names keep working through v1.x and are removed no earlier than v2.0.
"""
import warnings


class DeprecatedAttribute:
    """Class-level alias that forwards an old attribute name to its new name.

    Reading or writing the old name emits ``DeprecationWarning`` and acts on
    the new attribute, so both names always hold the same value.

    Parameters
    ----------
    new_name : str
        Name of the attribute that holds the value.

    Examples
    --------
    >>> class Model:
    ...     nat_freq = DeprecatedAttribute("natural_freq")
    """

    def __init__(self, new_name):
        self.new_name = new_name

    def __set_name__(self, owner, name):
        self.old_name = name

    def _warn(self):
        warnings.warn(
            f"{self.old_name} is deprecated; use {self.new_name} instead",
            DeprecationWarning,
            stacklevel=3,
        )

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        self._warn()
        return getattr(obj, self.new_name)

    def __set__(self, obj, value):
        self._warn()
        setattr(obj, self.new_name, value)

    def __delete__(self, obj):
        self._warn()
        delattr(obj, self.new_name)


def reject_both(old_name, new_name, new_value, new_default):
    """Raise when a renamed argument is passed under both names.

    Called after the deprecated name was found to be passed. The canonical
    argument counts as passed when it differs from its default.

    Parameters
    ----------
    old_name, new_name : str
        Deprecated and canonical argument names, used in the message.
    new_value : object
        Value of the canonical argument.
    new_default : object
        Default of the canonical argument.

    Raises
    ------
    TypeError
        When the canonical argument was passed as well.
    """
    if new_value is not new_default and new_value != new_default:
        raise TypeError(f"pass either {new_name} or {old_name}, not both")
