import { Button } from '@/components/ui/button.tsx';
import useClientIdentifier from '@/hooks/useClientIdentifier.tsx';
import { createAuthPin } from '@/services/PlexService.tsx';

const PLEX_PRODUCT_NAME = 'Plexio';

const Login = () => {
  const clientIdentifier = useClientIdentifier();

  const handleLogin = async () => {
    const { origin, pathname } = window.location;

    const authPin = await createAuthPin(clientIdentifier);
    const forwardUrl = new URL('/auth-redirect', origin);
    forwardUrl.searchParams.set('code', authPin.code);
    forwardUrl.searchParams.set('id', authPin.id);
    forwardUrl.searchParams.set('redirect', pathname);

    const loginParams = new URLSearchParams({
      clientID: clientIdentifier,
      code: authPin.code,
      forwardUrl: forwardUrl.toString(),
    });
    loginParams.set('context[device][product]', PLEX_PRODUCT_NAME);

    window.location.href = `https://app.plex.tv/auth#?${loginParams.toString()}`;
  };

  return (
    <div className="border rounded-lg p-6">
      <h1 className="text-xl font-bold text-center ">
        Plexio: Plex Interaction for Stremio
      </h1>
      <p className="text-sm text-center mt-2">
        Seamlessly connects your Plex and Stremio accounts, letting you enjoy
        your Plex media directly within Stremio.
      </p>
      <div className="mt-6">
        <Button
          onClick={() => {
            void handleLogin();
          }}
          className="w-full"
        >
          Login
        </Button>
      </div>
    </div>
  );
};

export default Login;
