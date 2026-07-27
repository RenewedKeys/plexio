from types import SimpleNamespace
from unittest import TestCase

from yarl import URL

from plexio.models.plex import PlexMediaMeta, PlexMediaType, Resolution


class TestPlexFilenameTagging(TestCase):
    def test_tag_filename_with_extension(self):
        tagged = PlexMediaMeta._tag_filename('Movie.Title.2024.mkv', 'Transcode')
        self.assertEqual(tagged, 'Movie.Title.2024 [Transcode].mkv')

    def test_tag_filename_without_extension(self):
        tagged = PlexMediaMeta._tag_filename('MovieTitle2024', 'Transcode')
        self.assertEqual(tagged, 'MovieTitle2024 [Transcode]')

    def test_transcode_paths_are_tagged_and_direct_play_unchanged(self):
        media = PlexMediaMeta(
            guid='imdb://tt1234567',
            type=PlexMediaType.movie,
            title='Movie',
            key='/library/metadata/100',
            librarySectionTitle='Movies',
            Media=[
                {
                    'videoResolution': '1080p',
                    'width': 1920,
                    'duration': 60000,
                    'Part': [
                        {
                            'file': '/media/movies/Movie.Title.2024.mkv',
                            'size': 12345,
                            'key': '/library/parts/1/file.mkv',
                            'Stream': [],
                        }
                    ],
                }
            ],
        )
        configuration = SimpleNamespace(
            server_name='Server',
            streaming_url=URL('http://example.com'),
            access_token='token',
            include_transcode_original=True,
            include_transcode_down=True,
            transcode_down_qualities=[Resolution.R720],
            include_plex_tv=False,
        )

        streams = media.get_stremio_streams(configuration)
        filenames = [stream.behavior_hints.filename for stream in streams]

        self.assertEqual(filenames[0], 'Movie.Title.2024.mkv')
        self.assertIn('Movie.Title.2024 [Transcode].mkv', filenames)
        self.assertIn('Movie.Title.2024 [Transcode-720p].mkv', filenames)
        self.assertNotEqual(
            'Movie.Title.2024 [Transcode].mkv',
            'Movie.Title.2024 [Transcode-720p].mkv',
        )
