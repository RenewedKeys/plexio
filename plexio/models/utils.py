import base64

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
    encoded_guid = plexio_id[len(PLEXIO_PREFIX) :]
    padding = (-len(encoded_guid)) % 4
    encoded_guid += '=' * padding
    return base64.urlsafe_b64decode(encoded_guid).decode()


def rating_key_to_plexio_id(rating_key: str | int) -> str:
    return f'{PLEXIO_RATING_KEY_PREFIX}{rating_key}'


def is_rating_key_plexio_id(plexio_id: str) -> bool:
    return plexio_id.startswith(PLEXIO_RATING_KEY_PREFIX)


def plexio_id_to_rating_key(plexio_id: str) -> str:
    return plexio_id[len(PLEXIO_RATING_KEY_PREFIX) :]
