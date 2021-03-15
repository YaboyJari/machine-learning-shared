const {
    getTensorsFromBagOfWords
} = require('./bag-of-words.js');
const tf = require('@tensorflow/tfjs');
const features = [
    'THIS IS A',
    'TEST TEST'
]

test('checks if bag of words works.', async () => {
    const bagOfWordsTensor = await getTensorsFromBagOfWords(features);
    return tf.tidy(() => {
        const testTensor = tf.tensor2d([
            [1, 1, 0, 1],
            [0, 0, 2, 0]
        ]);
        expect(bagOfWordsTensor.data).toBe(testTensor.data);
    });
});