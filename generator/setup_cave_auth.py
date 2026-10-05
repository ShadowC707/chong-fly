"""Interactive local CAVE credential setup; never put a token in CLI arguments.

Uses CAVEclient's native credential location. Saving is not access validation:
run generator.probe_cave --list-datastacks afterwards.
"""
import argparse
import getpass
import sys
import warnings


def setup_auth(*, replace=False, open_browser=True, auth_factory=None, read_secret=None):
    if auth_factory is None:
        from caveclient.auth import AuthClient
        auth_factory = AuthClient
    auth = auth_factory()
    if auth.token and not replace:
        return 'already_present'

    # Retrieve the existing-token page, not the token-creation endpoint:
    # creating a token can invalidate a token used on another computer.
    print('Sign in to your own CAVE account and copy the token locally:')
    print(auth.get_token_page(open=open_browser))
    print('Paste only into the hidden terminal prompt, not into chat or a source file.')
    if read_secret is None:
        read_secret = getpass.getpass
    with warnings.catch_warnings():
        # getpass otherwise falls back to visible input on unsupported terminals.
        warnings.simplefilter('error', getpass.GetPassWarning)
        token = read_secret('CAVE token (hidden): ').strip()
    if not token or any(character.isspace() for character in token):
        raise ValueError('Expected a non-empty single token')
    # One native location only; no synchronization or extra credential copies.
    auth.save_token(token=token, overwrite=replace, ignore_readonly=False,
                    write_to_server_file=False, local_server=False)
    return 'saved'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replace', action='store_true', help='Explicitly replace an existing local token')
    parser.add_argument('--no-browser', action='store_true', help='Print the token page without opening a browser')
    args = parser.parse_args(argv)
    if not sys.stdin.isatty():
        print('Run this command yourself in an interactive terminal; token input must be hidden.')
        return 2
    try:
        result = setup_auth(replace=args.replace, open_browser=not args.no_browser)
    except (KeyboardInterrupt, EOFError):
        print('\nSetup cancelled.')
        return 2
    except Exception as exc:
        # Library exception text could include credential data or local paths.
        print(f'Setup failed ({type(exc).__name__}); token contents were not printed.')
        return 2
    if result == 'already_present':
        print('A local token already exists; it was preserved. Presence does not prove access.')
        print('Use --replace only if you intend to replace that local credential.')
    else:
        print('Token saved locally. Access has not yet been verified.')
    print('Next: python -m generator.probe_cave --list-datastacks --output data/cave_datastacks.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
