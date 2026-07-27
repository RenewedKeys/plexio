from plexio.models.addon import AddonConfiguration
from plexio.models.plex import PlexMediaMeta, Resolution


def _make_media_item(width=1920, video_resolution='1080'):
    return {
        'guid': 'plex://movie/abcdef123456',
        'type': 'movie',
        'title': 'Test Movie',
        'librarySectionTitle': 'Movies',
        'key': '/library/metadata/1',
        'ratingKey': '1',
        'Media': [
            {
                'videoResolution': video_resolution,
                'width': width,
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


def test_is_4k_true_for_standard_uhd_width_and_label():
    assert PlexMediaMeta._is_4k({'width': 3840, 'videoResolution': '4k'}) is True


def test_is_4k_true_for_dci_cinema_width_with_mismatched_label():
    assert PlexMediaMeta._is_4k({'width': 4096, 'videoResolution': '1080'}) is True


def test_is_4k_true_from_label_when_width_missing():
    assert PlexMediaMeta._is_4k({'width': None, 'videoResolution': '4k'}) is True
    assert PlexMediaMeta._is_4k({'videoResolution': '4k'}) is True


def test_is_4k_false_for_1080p():
    assert PlexMediaMeta._is_4k({'width': 1920, 'videoResolution': '1080'}) is False


def test_is_4k_false_for_720p():
    assert PlexMediaMeta._is_4k({'width': 1280, 'videoResolution': '720'}) is False


def test_get_stremio_streams_skips_transcode_for_4k_source():
    meta = PlexMediaMeta.model_validate(
        _make_media_item(width=3840, video_resolution='4k'),
    )
    configuration = _make_configuration()

    streams = meta.get_stremio_streams(configuration)

    assert len(streams) == 1
    quality_description = streams[0].behavior_hints.binge_group
    assert quality_description.startswith('Direct Play')


def test_get_stremio_streams_offers_transcode_for_non_4k_source():
    meta = PlexMediaMeta.model_validate(
        _make_media_item(width=1920, video_resolution='1080'),
    )
    configuration = _make_configuration()

    streams = meta.get_stremio_streams(configuration)

    assert len(streams) == 3
    direct_play, transcode_original, transcode_down = streams
    assert direct_play.behavior_hints.binge_group.startswith('Direct Play')
    assert transcode_original.behavior_hints.binge_group.startswith('Transcode')
    assert transcode_down.behavior_hints.binge_group.startswith('Transcode')
