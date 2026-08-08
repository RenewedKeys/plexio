import { FC, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import Loading from '@/components/loading.tsx';
import useClientIdentifier from '@/hooks/useClientIdentifier.tsx';
import { SetPlexToken } from '@/hooks/usePlexToken.tsx';
import { getAuthToken } from '@/services/PlexService.tsx';

interface Props {
  setPlexToken: SetPlexToken;
}

const AuthRedirectPage: FC<Props> = ({ setPlexToken }) => {
  const [searchParams] = useSearchParams();
  const clientIdentifier = useClientIdentifier();
  const navigate = useNavigate();

  useEffect(() => {
    if (!clientIdentifier) return;

    const id = searchParams.get('id');
    const code = searchParams.get('code');
    const redirect = searchParams.get('redirect');
    const safeRedirect =
      redirect?.startsWith('/') &&
      !redirect.startsWith('//') &&
      !redirect.includes('\\')
        ? redirect
        : '/';

    const setAuthToken = async (): Promise<void> => {
      if (!id || !code) {
        void navigate('/', { replace: true });
        return;
      }
      const authToken = await getAuthToken({ id, code }, clientIdentifier);
      setPlexToken(authToken);
      void navigate(safeRedirect, { replace: true });
    };

    void setAuthToken();
  }, [searchParams, clientIdentifier, navigate, setPlexToken]);

  return <Loading />;
};

export default AuthRedirectPage;
