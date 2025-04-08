import numpy as np

# Define the emotion mapping
EMOTION_MAPPING = {
    'Neutral': 0,
    'Confusion': 1,
    'Anger': 2,
    'Disgust': 3,
    'Frustration': 4,
    'Sadness': 5,
    'Surprise': 6,
    'Joy': 7,
    'Fear': 8
}

# Reverse mapping for decoding
EMOTION_REVERSE_MAPPING = {v: k for k, v in EMOTION_MAPPING.items()}

def encode_emotion(emotion):
    """
    Encode an emotion string to its corresponding numeric value.
    
    Args:
        emotion (str): The emotion label (e.g., 'Joy', 'Anger', etc.)
        
    Returns:
        int: The numeric encoding of the emotion
        
    Raises:
        ValueError: If the emotion is not in the mapping
    """
    if emotion not in EMOTION_MAPPING:
        raise ValueError(f"Unknown emotion: {emotion}. Valid emotions are: {list(EMOTION_MAPPING.keys())}")
    
    return EMOTION_MAPPING[emotion]

def decode_emotion(encoded_value):
    """
    Decode a numeric value back to its corresponding emotion string.
    
    Args:
        encoded_value (int): The numeric encoding of the emotion
        
    Returns:
        str: The emotion label
        
    Raises:
        ValueError: If the encoded value is not in the reverse mapping
    """
    if encoded_value not in EMOTION_REVERSE_MAPPING:
        raise ValueError(f"Unknown encoded value: {encoded_value}. Valid values are: {list(EMOTION_REVERSE_MAPPING.keys())}")
    
    return EMOTION_REVERSE_MAPPING[encoded_value]

def encode_emotions(emotions):
    """
    Encode a list of emotions to their corresponding numeric values.
    
    Args:
        emotions (list): List of emotion strings
        
    Returns:
        numpy.ndarray: Array of encoded emotion values
    """
    return np.array([encode_emotion(emotion) for emotion in emotions])

def decode_emotions(encoded_values):
    """
    Decode a list of numeric values back to their corresponding emotion strings.
    
    Args:
        encoded_values (numpy.ndarray): Array of encoded emotion values
        
    Returns:
        list: List of emotion strings
    """
    return [decode_emotion(value) for value in encoded_values]

def get_num_classes():
    """
    Get the number of emotion classes.
    
    Returns:
        int: Number of emotion classes
    """
    return len(EMOTION_MAPPING)

def get_emotion_names():
    """
    Get a list of all emotion names.
    
    Returns:
        list: List of emotion names
    """
    return list(EMOTION_MAPPING.keys()) 