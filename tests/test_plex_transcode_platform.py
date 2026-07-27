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


def test_transcode_urls_omit_x_plex_platform():
    meta = PlexMediaMeta.model_validate(_make_media_item())
    configuration = _make_configuration()

    streams = meta.get_stremio_streams(configuration)

    _, transcode_original, transcode_down = streams[:3]

    for stream in (transcode_original, transcode_down):
        assert 'X-Plex-Platform' not in stream.url


def test_transcode_urls_retain_other_expected_params():
    meta = PlexMediaMeta.model_validate(_make_media_item())
    configuration = _make_configuration()

    streams = meta.get_stremio_streams(configuration)

    _, transcode_original, transcode_down = streams[:3]

    for stream in (transcode_original, transcode_down):
        assert 'protocol=hls' in stream.url
        assert 'X-Plex-Token=test-token' in stream.url
        assert 'path=/library/metadata/1' in stream.url
        assert 'fastSeek=1' in stream.url
        assert 'copyts=1' in stream.url
        assert 'autoAdjustQuality=0' in stream.url
