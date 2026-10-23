from types import SimpleNamespace

import pytest


def test_setup_preserves_existing_token_without_prompting():
    from generator.setup_cave_auth import setup_auth
    auth = SimpleNamespace(token='EXISTING_SECRET')
    result = setup_auth(auth_factory=lambda: auth,
                        read_secret=lambda _: pytest.fail('Must not request a replacement'))
    assert result == 'already_present'


def test_setup_saves_hidden_input_to_one_native_location_without_echo(capsys):
    from generator.setup_cave_auth import setup_auth
    saved = []
    auth = SimpleNamespace(token=None,
        get_token_page=lambda open: 'https://example.test/auth/tokens',
        save_token=lambda **kwargs: saved.append(kwargs))
    assert setup_auth(auth_factory=lambda: auth, read_secret=lambda _: '  TEST_SECRET  ') == 'saved'
    assert saved == [dict(token='TEST_SECRET', overwrite=False, ignore_readonly=False,
                          write_to_server_file=False, local_server=False)]
    output = capsys.readouterr()
    assert 'TEST_SECRET' not in output.out + output.err


@pytest.mark.parametrize('value', ['', '   ', 'part1\npart2'])
def test_setup_rejects_empty_or_multiline_input(value):
    from generator.setup_cave_auth import setup_auth
    auth = SimpleNamespace(token=None, get_token_page=lambda open: 'https://example.test',
                           save_token=lambda **kwargs: pytest.fail('Must not save invalid input'))
    with pytest.raises(ValueError):
        setup_auth(auth_factory=lambda: auth, read_secret=lambda _: value)


def test_setup_replacement_requires_explicit_flag():
    from generator.setup_cave_auth import setup_auth
    saved = []
    auth = SimpleNamespace(token='OLD', get_token_page=lambda open: 'https://example.test',
                           save_token=lambda **kwargs: saved.append(kwargs))
    assert setup_auth(replace=True, auth_factory=lambda: auth, read_secret=lambda _: 'NEW') == 'saved'
    assert saved[0]['overwrite'] is True


def test_hidden_input_failure_does_not_fall_back_to_visible_input():
    from generator.setup_cave_auth import setup_auth
    import getpass
    import warnings
    auth = SimpleNamespace(token=None, get_token_page=lambda open: 'https://example.test',
                           save_token=lambda **kwargs: pytest.fail('Must not save after input failure'))
    def unsupported_terminal(_):
        warnings.warn('Cannot hide input', getpass.GetPassWarning)
        pytest.fail('Visible fallback must never be reached')
    with pytest.raises(getpass.GetPassWarning):
        setup_auth(auth_factory=lambda: auth, read_secret=unsupported_terminal)


def test_cli_refuses_noninteractive_token_input(monkeypatch, capsys):
    from generator import setup_cave_auth
    monkeypatch.setattr(setup_cave_auth.sys, 'stdin', SimpleNamespace(isatty=lambda: False))
    monkeypatch.setattr(setup_cave_auth, 'setup_auth', lambda **kwargs: pytest.fail('No credential access'))
    assert setup_cave_auth.main([]) == 2
    assert 'interactive terminal' in capsys.readouterr().out


def test_native_auth_save_and_reload_stays_in_explicit_temporary_file(tmp_path, monkeypatch, capsys):
    pytest.importorskip('caveclient')
    from caveclient.auth import AuthClient
    from generator.setup_cave_auth import setup_auth
    token_file = tmp_path / 'test-credential.json'
    auth = AuthClient(token_file=str(token_file), token='')
    monkeypatch.setattr(auth, 'get_token_page', lambda open: 'https://example.test')
    assert setup_auth(auth_factory=lambda: auth, read_secret=lambda _: 'FAKE_LOCAL_TEST_VALUE') == 'saved'
    assert AuthClient(token_file=str(token_file)).token == 'FAKE_LOCAL_TEST_VALUE'
    assert list(tmp_path.iterdir()) == [token_file]
    assert 'FAKE_LOCAL_TEST_VALUE' not in capsys.readouterr().out
