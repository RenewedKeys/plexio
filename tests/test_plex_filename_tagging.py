from plexio.models.addon import AddonConfiguration
from plexio.models.plex import PlexMediaMeta, Resolution


def _make_media_item():
    return {
        'guid': 'plex://movie/abcdef123456',
        'type': 'movie',
        'title': 'Test Movie',
        'librarySectionTitle': 'Movies',
        'key': '/library/metadata/1',
        'ratingKey': '1',
        'Media': [
            {
                'videoResolution': '1080',
                'width': 1920,
                'Part': [
                    {
                        'file': '/data/Movies/Test Movie (2024)/Test.Movie.2024.mkv',
                        'size': 123456789,
                        'key': '/library/parts/1/1234567890/file.mkv',
                        'Stream': [],
                    },
                ],
            },
        ],
    }


def _make_configuration(**overrides):
    fields = {
        'accessToken': 'test-token',
        'discoveryUrl': 'http://plex.example.com:32400',
        'streamingUrl': 'http://plex.example.com:32400',
        'serverName': 'TestServer',
        'includeTranscodeOriginal': True,
        'includeTranscodeDown': True,
        'transcodeDownQualities': [Resolution.R720],
        **overrides,
    }
    return AddonConfiguration(**fields)


def test_tag_filename_with_extension():
    result = PlexMediaMeta._tag_filename('Movie.Title.2024.mkv', 'Transcode')
    assert result == 'Movie.Title.2024 [Transcode].mkv'


def test_tag_filename_without_extension():
    result = PlexMediaMeta._tag_filename('Movie Title 2024', 'Transcode')
    assert result == 'Movie Title 2024 [Transcode]'


def test_get_stremio_streams_tags_transcode_filenames_only():
    meta = PlexMediaMeta.model_validate(_make_media_item())
    configuration = _make_configuration()

    streams = meta.get_stremio_streams(configuration)

    direct_play, transcode_original, transcode_down = streams[:3]

    original_filename = 'Test.Movie.2024.mkv'
    assert direct_play.behavior_hints.filename == original_filename

    assert transcode_original.behavior_hints.filename == (
        'Test.Movie.2024 [Transcode].mkv'
    )
    assert '[Transcode]' in transcode_original.behavior_hints.filename
    assert transcode_original.behavior_hints.filename != original_filename

    assert transcode_down.behavior_hints.filename == (
        'Test.Movie.2024 [Transcode-720p].mkv'
    )
    assert '[Transcode-720p]' in transcode_down.behavior_hints.filename
    assert transcode_down.behavior_hints.filename != original_filename
    assert (
        transcode_down.behavior_hints.filename
        != transcode_original.behavior_hints.filename
    )
