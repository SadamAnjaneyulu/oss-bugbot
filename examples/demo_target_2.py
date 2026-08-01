"""Second demo target - CLI-mode positive control.

Different bug category from examples/demo_target.py's null-deref (logic):
this one is a SQL injection (security), to test cli.py's code path
specifically against a bug class the earlier Actions-mode test didn't cover.
"""


def get_user(user_id, db):
    query = f"SELECT * FROM users WHERE id = {user_id}"
    return db.execute(query)
