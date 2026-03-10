#!/usr/bin/env python3
"""CLI tool for managing user accounts.

Usage:
    python manage_users.py create <username> <password> [--role admin|user] [--display-name "Name"]
    python manage_users.py list
    python manage_users.py delete <username>
    python manage_users.py reset-password <username> <new_password>
"""

import argparse
import sys
import os

# Allow running from the Server directory or project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from UserDB import UserDB


def main():
    parser = argparse.ArgumentParser(description="Manage user accounts for LogFileAnalyzer")
    parser.add_argument("--db", default=None, help="Path to SQLite database file (default: DATA_BASE_DIRECTORY/users.db)")
    sub = parser.add_subparsers(dest="command", required=True)

    # create
    p_create = sub.add_parser("create", help="Create a new user")
    p_create.add_argument("username")
    p_create.add_argument("password")
    p_create.add_argument("--role", default="user", choices=["user", "admin"])
    p_create.add_argument("--display-name", default=None)

    # list
    sub.add_parser("list", help="List all users")

    # delete
    p_del = sub.add_parser("delete", help="Delete a user")
    p_del.add_argument("username")

    # reset-password
    p_rp = sub.add_parser("reset-password", help="Reset a user's password")
    p_rp.add_argument("username")
    p_rp.add_argument("new_password")

    args = parser.parse_args()
    db = UserDB(args.db)

    if args.command == "create":
        try:
            user = db.create_user(
                args.username,
                args.password,
                display_name=args.display_name,
                role=args.role,
            )
            print(f"Created user: id={user['id']}  username={user['username']}  role={user['role']}")
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.command == "list":
        users = db.list_users()
        if not users:
            print("No users found.")
        else:
            fmt = "{:<6} {:<20} {:<20} {:<8} {}"
            print(fmt.format("ID", "Username", "Display Name", "Role", "Created"))
            print("-" * 80)
            for u in users:
                print(fmt.format(u["id"], u["username"], u["display_name"] or "", u["role"], u["created_at"]))

    elif args.command == "delete":
        if db.delete_user(args.username):
            print(f"Deleted user '{args.username}'")
        else:
            print(f"User '{args.username}' not found", file=sys.stderr)
            sys.exit(1)

    elif args.command == "reset-password":
        if db.reset_password(args.username, args.new_password):
            print(f"Password reset for '{args.username}'")
        else:
            print(f"User '{args.username}' not found", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
