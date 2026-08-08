import { useEffect, useState } from 'react';
import { PlexToken } from '@/hooks/usePlexToken.tsx';
import { getSections } from '@/services/PMSService.tsx';
import { PlexSection } from '@/types/plex.tsx';

const usePMSSections = (serverUrl: string, plexToken: PlexToken) => {
  const [sections, setSections] = useState<PlexSection[]>([]);

  useEffect(() => {
    setSections([]);
    if (!plexToken || !serverUrl) {
      return;
    }

    const fetchSections = async (): Promise<void> => {
      const sectionsData = await getSections(serverUrl, plexToken);
      setSections(sectionsData);
    };

    void fetchSections();
  }, [serverUrl, plexToken]);

  return sections;
};

export default usePMSSections;
