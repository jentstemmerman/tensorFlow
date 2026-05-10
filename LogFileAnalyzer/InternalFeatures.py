import tensorflow as tf


INTERNAL_EOL_TOKEN = "__INTERNAL_EOL__"
INTERNAL_EOL_LAYER_NAME = "Layer_Internal_Eol_Cathegorical_layer"


def append_internal_features(token_list):
    """Append all synthetic/internal features to one parsed token row."""
    return token_list + [INTERNAL_EOL_TOKEN]


def append_internal_features_to_rows(token_list_list):
    """Append synthetic/internal features to all parsed rows."""
    return [append_internal_features(row) for row in token_list_list]


def get_internal_feature_count(item_list):
    return len(item_list) + 1


def is_internal_feature_index(feature_index, item_list):
    return feature_index >= len(item_list)


def create_internal_eol_lookup(mask_token):
    return tf.keras.layers.StringLookup(vocabulary=[INTERNAL_EOL_TOKEN], mask_token=mask_token)


def create_internal_eol_inverse_lookup(mask_token):
    return tf.keras.layers.StringLookup(vocabulary=[INTERNAL_EOL_TOKEN], mask_token=mask_token, invert=True)


def is_internal_eol_lookup(lookup_layer):
    if lookup_layer is None:
        return False

    try:
        vocabulary = lookup_layer.get_vocabulary()
    except Exception:
        return False

    return INTERNAL_EOL_TOKEN in vocabulary