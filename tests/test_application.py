from unittest import TestCase

from plexio.main import app


class ApplicationTests(TestCase):
    def test_application_registers_core_and_proxy_routes(self):
        paths = {route.path for route in app.routes}
        self.assertTrue(
            {
                '/manifest.json',
                '/api/v1/sessions',
                '/api/v1/plex-pin',
                '/api/v1/plex-token/{pin_id}',
                '/api/v1/plex-resources',
            }.issubset(paths)
        )

    def test_playback_routes_accept_get_and_head(self):
        play_routes = [
            route
            for route in app.routes
            if route.path == '/{session_id}/play/{rating_key}/{duration}/{part_b64}'
        ]
        self.assertEqual(len(play_routes), 1)
        self.assertTrue({'GET', 'HEAD'}.issubset(play_routes[0].methods))
