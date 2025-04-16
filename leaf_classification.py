# import numpy as np
# from skimage import io, filters, transform
# from skimage.color import rgb2gray, rgba2rgb
# import os
# from collections import Counter
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.preprocessing import StandardScaler

# def extract_features(image_path, resize_shape=(128, 128)):
#     """
#     Extract features from an image using Sobel edges and multiple Gabor filters.
#     Features include the mean and standard deviation of Sobel edge responses as well
#     as the means and standard deviations of several Gabor filter responses.
#     """
#     image = io.imread(image_path)
#     if len(image.shape) == 3:
#         if image.shape[2] == 4:  # RGBA image
#             image = rgba2rgb(image)
#         image = rgb2gray(image)
#     elif len(image.shape) == 2:
#         pass
#     else:
#         raise ValueError(f"Unexpected image shape: {image.shape}")

#     image = transform.resize(image, resize_shape, anti_aliasing=True)
    
#     sobel_edges = filters.sobel(image)
#     edge_mean = np.mean(sobel_edges)
#     edge_std = np.std(sobel_edges)
    
#     gabor_features = []
#     frequencies = [0.1, 0.3, 0.5]
#     thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
#     for freq in frequencies:
#         for theta in thetas:
#             gabor_response = filters.gabor(image, frequency=freq, theta=theta)[0]
#             gabor_features.append(np.mean(gabor_response))
#             gabor_features.append(np.std(gabor_response))
    
#     features = [edge_mean, edge_std] + gabor_features
#     return np.array(features)

# def load_dataset(data_dir):
#     """
#     Load dataset by iterating over species directories contained within data_dir.
#     It sorts the directory names so the species-to-index mapping is consistent.
#     Returns:
#       - X: Array of feature vectors.
#       - y: Array of labels corresponding to each feature vector.
#       - species: List of species names in sorted order.
#     Also prints sample counts for each species.
#     """
#     X, y = [], []
#     species = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
#     sample_counts = {}
#     for i, species_name in enumerate(species):
#         species_path = os.path.join(data_dir, species_name)
#         image_names = [img for img in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, img))]
#         sample_counts[species_name] = len(image_names)
#         if image_names:
#             for image_name in image_names:
#                 image_path = os.path.join(species_path, image_name)
#                 try:
#                     features = extract_features(image_path)
#                     X.append(features)
#                     y.append(i)
#                 except Exception as e:
#                     print(f"Error processing {image_path}: {e}")
    
#     print("Sample counts per species:", sample_counts)
#     return np.array(X), np.array(y), species

# if __name__ == "__main__":
#     data_dir = "leaf_dataset"
    
#     X, y, species_names = load_dataset(data_dir)
    
#     print("\nFeature summaries by species:")
#     for idx, name in enumerate(species_names):
#         species_features = X[y == idx]
#         print(f"Species: {name}")
#         print("Mean of features:", np.mean(species_features, axis=0))
#         print("Std of features:", np.std(species_features, axis=0), "\n")
    
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
#     scaler = StandardScaler()
#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)
    
#     clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
#     clf.fit(X_train, y_train)
    
#     y_pred = clf.predict(X_test)
#     accuracy = accuracy_score(y_test, y_pred)
#     print(f"Accuracy: {accuracy:.2f}")
    

#     new_image_path = "boss.png"
    
#     try:
#         if not os.path.isfile(new_image_path):
#             raise FileNotFoundError(f"Image file not found: {new_image_path}")
#         new_features = extract_features(new_image_path)
#         new_features = scaler.transform([new_features])
#         predicted_species = clf.predict(new_features)[0]
#         print(f"Predicted species: {species_names[predicted_species]}")
#     except FileNotFoundError as e:
#         print(e)
#     except Exception as e:
#         print(f"Error processing new image {new_image_path}: {e}")
import numpy as np
from skimage import io, filters, transform
from skimage.color import rgb2gray, rgba2rgb
import os
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

def extract_features(image_path, resize_shape=(128, 128)):
    """
    Extract features from an image using Sobel edges and multiple Gabor filters.
    Features include the mean and standard deviation of Sobel edge responses as well
    as the means and standard deviations of several Gabor filter responses.
    """
    image = io.imread(image_path)
    if len(image.shape) == 3:
        if image.shape[2] == 4:  
            image = rgba2rgb(image)
        image = rgb2gray(image)
    elif len(image.shape) == 2:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    image = transform.resize(image, resize_shape, anti_aliasing=True)
    
    sobel_edges = filters.sobel(image)
    edge_mean = np.mean(sobel_edges)
    edge_std = np.std(sobel_edges)
    
    gabor_features = []
    frequencies = [0.1, 0.3, 0.5]
    thetas = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for freq in frequencies:
        for theta in thetas:
            response, _ = filters.gabor(image, frequency=freq, theta=theta)
            gabor_features.append(np.mean(response))
            gabor_features.append(np.std(response))
    
    features = np.array([edge_mean, edge_std] + gabor_features)
    
    if np.isnan(features).any():
        raise ValueError(f"NaN values encountered in features for image: {image_path}")
    
    return features

def load_dataset(data_dir):
    """
    Load dataset by iterating over species directories within data_dir.
    The directories are sorted for a consistent species-to-index mapping.
    Only images that yield valid features (without NaN values) are included.
    """
    X, y = [], []
    species = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    sample_counts = {}
    valid_counts = {}
    
    for i, species_name in enumerate(species):
        species_path = os.path.join(data_dir, species_name)
        image_names = [img for img in os.listdir(species_path) if os.path.isfile(os.path.join(species_path, img))]
        sample_counts[species_name] = len(image_names)
        valid_counts[species_name] = 0
        
        if image_names:
            for image_name in image_names:
                image_path = os.path.join(species_path, image_name)
                try:
                    features = extract_features(image_path)
                    X.append(features)
                    y.append(i)
                    valid_counts[species_name] += 1
                except Exception as e:
                    print(f"Skipping {image_path}: {e}")
    
    print("Total samples per species:", sample_counts)
    print("Valid samples per species (after filtering):", valid_counts)
    
    return np.array(X), np.array(y), species

if __name__ == "__main__":
    data_dir = "leaf_dataset"
    
    X, y, species_names = load_dataset(data_dir)
    
    print("\nFeature summaries by species:")
    for idx, name in enumerate(species_names):
        species_features = X[y == idx]
        if species_features.size == 0:
            print(f"Species: {name} has no valid features.\n")
            continue
        print(f"Species: {name}")
        print("Mean of features:", np.mean(species_features, axis=0))
        print("Std of features:", np.std(species_features, axis=0), "\n")
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on test set: {accuracy:.2f}")
    
    new_image_path = "boss.png"
    try:
        if not os.path.isfile(new_image_path):
            raise FileNotFoundError(f"Image file not found: {new_image_path}")
        new_features = extract_features(new_image_path)
        new_features = scaler.transform([new_features])
        predicted_species = clf.predict(new_features)[0]
        print(f"Predicted species for new image: {species_names[predicted_species]}")
    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"Error processing new image {new_image_path}: {e}")
