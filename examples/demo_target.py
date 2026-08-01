"""Demo target for verifying the bot end to end on a real PR.

This file exists only so review.yml has something to run against on its
first live trigger. Not part of the bot's own source (src/).
"""


def get_display_name(user_id, users_by_id):
    user = users_by_id.get(user_id)
    return user.name.strip()
