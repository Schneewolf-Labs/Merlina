"""Operator-only account creation and consistent SQLite backup."""
import argparse
import getpass
import json
from .settings import HostedSettings
from .store import HostedStore


def main():
    parser = argparse.ArgumentParser(description='Merlina hosted pilot administration')
    parser.add_argument('--database', help='Hosted SQLite database path')
    commands = parser.add_subparsers(dest='command', required=True)
    account = commands.add_parser('create-account')
    account.add_argument('email')
    backup = commands.add_parser('backup')
    backup.add_argument('destination')
    args = parser.parse_args()
    settings = HostedSettings.from_env()
    if args.database:
        settings.database_path = args.database
    store = HostedStore(settings.database_path)
    if args.command == 'create-account':
        password = getpass.getpass('Password (12+ characters): ')
        if password != getpass.getpass('Confirm password: '):
            parser.error('Passwords do not match')
        print(json.dumps(store.create_account(args.email, password)))
    else:
        store.backup(args.destination)
        print('Backup created and integrity verified')


if __name__ == '__main__':
    main()
