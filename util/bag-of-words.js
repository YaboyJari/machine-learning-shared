require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');

const intersectArrayIndex = (sentences, voc) => {
    let sortedSentence = sentences.concat().sort();
    let sortedVoc = voc.concat().sort();
    let indexCounter = [];
    let sentenceIndex = 0;
    let vocIndex = 0;

    while (sentenceIndex < sentences.length &&
        vocIndex < voc.length) {
        if (sortedSentence[sentenceIndex] === sortedVoc[vocIndex]) {
            indexCounter.push(vocIndex);
            sentenceIndex++;
        } else if (sortedSentence[sentenceIndex] < sortedVoc[vocIndex]) {
            sentenceIndex++;
        } else {
            vocIndex++;
        }
    };
    return indexCounter;
};

const buildVoc = (features) => {
    let wordArray = [];
    features.forEach(feature => {
        const words = feature.split(' ');
        wordArray = wordArray.concat(words);
    });

    const vocSet = new Set(wordArray);

    const vocArray = Array.from(vocSet).sort();

    return vocArray;
};

const searchForWordsAndReturnIndex = (features, voc) => {
    const indexesArray = [];

    features.forEach(feature => {
        const words = feature.split(' ');
        const indexes = intersectArrayIndex(words, voc);
        indexesArray.push(indexes);
    });
    return indexesArray;
};

const countOccurrences = (arr, val) => arr.reduce((a, v) => (v === val ? a + 1 : a), 0);

const countMultiplesAndAddToArray = (feature) => {
    let multipleArray = [];
    feature.forEach(number => {
        multipleArray.push([number, countOccurrences(feature, number)]);
    });

    const stringArray = multipleArray.map(JSON.stringify);
    const uniqueStringArray = new Set(stringArray);
    const uniqueArray = Array.from(uniqueStringArray, JSON.parse);

    return uniqueArray;
};

const setSparseToDenseTensor = (featureIndexes, voc) => {
    const indicesArray = [];
    const valueArray = [];

    featureIndexes.forEach((feature, index) => {
        feature = countMultiplesAndAddToArray(feature);
        feature.forEach(data => {
            indicesArray.push([index, data[0]]);
            valueArray.push(data[1]);
        });
    });

    const values = tf.tensor1d(valueArray, 'float32');
    const shape = [featureIndexes.length, voc.length];
    if (!indicesArray || !valueArray) {
        return new Error('Fault with indices or values');
    };

    return tf.sparseToDense(indicesArray, values, shape);
};

// this expects features as a sentences array
const getTensorsFromBagOfWords = async (features, voc) => {
    if (!voc) {
        voc = buildVoc(features);
    };
    const indexesOfFeatures = searchForWordsAndReturnIndex(features, voc);
    return transferToTensorData(indexesOfFeatures, voc);
};

const transferToTensorData = (features, voc) => {
    return tf.tidy(() => {
        try {
            features = setSparseToDenseTensor(features, voc);
            return features;
        } catch (err) {
            return err;
        };
    });
};

module.exports = {
    getTensorsFromBagOfWords,
}