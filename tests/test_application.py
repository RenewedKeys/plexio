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
