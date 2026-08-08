import base64
import binascii

LANGUAGE_TO_EMOJI = {
    'ps': '🇵🇰',
    'uz': '🇺🇿',
    'tk': '🇹🇲',
    'sq': '🇦🇱',
    'ar': '🇦🇪',
    'en': '🇬🇧',
    'sm': '🇼🇸',
    'ca': '🏴󠁥󠁳󠁣󠁴󠁿',
    'pt': '🇵🇹',
    'es': '🇪🇸',
    'gn': '🇵🇾',
    'hy': '🇦🇲',
    'ru': '🇷🇺',
    'nl': '🇳🇱',
    'pa': '🇮🇳',
    'de': '🇩🇪',
    'az': '🇦🇿',
    'bn': '🇧🇩',
    'be': '🇧🇾',
    'fr': '🇫🇷',
    'dz': '🇧🇹',
    'ay': '🇧🇴',
    'qu': '🇧🇴',
    'bs': '🇧🇦',
    'hr': '🇭🇷',
    'sr': '🇷🇸',
    'tn': '🇹🇳',
    'no': '🇧🇻',
    'nb': '🇧🇻',
    'nn': '🇧🇻',
    'ms': '🇲🇾',
    'bg': '🇧🇬',
    'ff': '🇸🇳',
    'rn': '🇧🇮',
    'km': '🇰🇭',
    'sg': '🇨🇫',
    'zh': '🇨🇳',
    'ln': '🇨🇩',
    'kg': '🇨🇬',
    'sw': '🇹🇿',
    'lu': '🇨🇩',
    'el': '🇬🇷',
    'tr': '🇹🇷',
    'cs': '🇨🇿',
    'sk': '🇸🇰',
    'da': '🇩🇰',
    'ti': '🇪🇷',
    'et': '🇪🇪',
    'ss': '🇸🇿',
    'am': '🇪🇹',
    'fo': '🇫🇴',
    'fj': '🇫🇯',
    'hi': '🇮🇳',
    'ur': '🇵🇰',
    'fi': '🇫🇮',
    'sv': '🇸🇪',
    'ka': '🇬🇪',
    'kl': '🇬🇱',
    'ch': '🇬🇺',
    'ht': '🇭🇹',
    'it': '🇮🇹',
    'la': '🇻🇦',
    'hu': '🇭🇺',
    'is': '🇮🇸',
    'id': '🇮🇩',
    'fa': '🇮🇷',
    'ku': '🇮🇶',
    'ga': '🇮🇪',
    'gv': '🇮🇲',
    'he': '🇮🇱',
    'ja': '🇯🇵',
    'kk': '🇰🇿',
    'ko': '🇰🇷',
    'ky': '🇰🇬',
    'lo': '🇱🇦',
    'lv': '🇱🇻',
    'st': '🇱🇸',
    'lt': '🇱🇹',
    'lb': '🇱🇺',
    'mg': '🇲🇬',
    'ny': '🇲🇼',
    'dv': '🇲🇻',
    'mt': '🇲🇹',
    'mh': '🇲🇭',
    'ro': '🇲🇩',
    'mn': '🇲🇳',
    'my': '🇲🇲',
    'af': '🇳🇦',
    'na': '🇳🇷',
    'ne': '🇳🇵',
    'mi': '🇳🇿',
    'mk': '🇲🇰',
    'pl': '🇵🇱',
    'rw': '🇷🇼',
    'ta': '🇮🇳',
    'sl': '🇸🇮',
    'so': '🇸🇴',
    'nr': '🇿🇦',
    'ts': '🇿🇦',
    've': '🇿🇦',
    'xh': '🇿🇦',
    'zu': '🇿🇦',
    'eu': '🇪🇸',
    'gl': '🇪🇸',
    'oc': '🇪🇸',
    'si': '🇱🇰',
    'tg': '🇹🇯',
    'th': '🇹🇭',
    'to': '🇹🇴',
    'uk': '🇺🇦',
    'bi': '🇻🇺',
    'vi': '🇻🇳',
    'sn': '🇿🇼',
    'nd': '🇿🇦',
}

PLEXIO_PREFIX = 'plexio:'
PLEXIO_RATING_KEY_PREFIX = 'plexio:rk-'


def get_flag_emoji(code):
    return LANGUAGE_TO_EMOJI.get(code, code)


def to_camel(string: str) -> str:
    words = string.split('_')
    return words[0].lower() + ''.join(word.capitalize() for word in words[1:])


def guid_to_plexio_id(guid: str) -> str:
    encoded_guid = base64.urlsafe_b64encode(guid.encode()).rstrip(b'=').decode()
    return PLEXIO_PREFIX + encoded_guid


def plexio_id_to_guid(plexio_id: str) -> str:
    if not plexio_id.startswith(PLEXIO_PREFIX):
        raise ValueError('Invalid Plexio GUID id')
    encoded_guid = plexio_id[len(PLEXIO_PREFIX) :]
    if not encoded_guid:
        raise ValueError('Invalid Plexio GUID id')
    padding = (-len(encoded_guid)) % 4
    encoded_guid += '=' * padding
    try:
        decoded = base64.b64decode(encoded_guid, altchars=b'-_', validate=True)
        return decoded.decode()
    except (binascii.Error, UnicodeDecodeError) as exc:
        raise ValueError('Invalid Plexio GUID id') from exc


def rating_key_to_plexio_id(rating_key: str | int) -> str:
    value = str(rating_key)
    if not value.isascii() or not value.isdigit():
        raise ValueError('Plex rating keys must be numeric')
    return f'{PLEXIO_RATING_KEY_PREFIX}{value}'


def is_rating_key_plexio_id(plexio_id: str) -> bool:
    if not plexio_id.startswith(PLEXIO_RATING_KEY_PREFIX):
        return False
    value = plexio_id[len(PLEXIO_RATING_KEY_PREFIX) :]
    parts = value.split(':')
    return len(parts) in (1, 3) and all(
        part.isascii() and part.isdigit() for part in parts
    )


def plexio_id_to_rating_key(plexio_id: str) -> str:
    if not is_rating_key_plexio_id(plexio_id):
        raise ValueError('Invalid Plexio rating-key id')
    return plexio_id[len(PLEXIO_RATING_KEY_PREFIX) :]
