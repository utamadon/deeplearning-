from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
from tensorflow.keras import layers

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print('-----------gpuメモリ------------------------------------------')
    print(e)
    print('-----------------------------------------------------')




classifier_url ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2" #@param {type:"string"}

IMAGE_SHAPE = (224, 224)

classifier = tf.keras.Sequential([
    hub.KerasLayer(classifier_url, input_shape=IMAGE_SHAPE+(3,))
])

import numpy as np
import PIL.Image as Image

grace_hopper = tf.keras.utils.get_file('image.jpg','https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg')
grace_hopper = Image.open(grace_hopper).resize(IMAGE_SHAPE)
print(grace_hopper)

#PILのやつで画像を書き込む
#------------------------------------------------------------------------------
print(grace_hopper.format, grace_hopper.size, grace_hopper.mode)
grace_hopper.show()
grace_hopper.save('images/lenna_square_pillow.jpg', quality=95)
#------------------------------------------------------------------------------

grace_hopper = np.array(grace_hopper)/255.0
grace_hopper.shape

#バッチの次元を追加し、画像をモデルに入力します。
result = classifier.predict(grace_hopper[np.newaxis, ...])
print('バッチの次元を追加し、画像をモデルに入力-------------------------------')
print(result.shape)
    
#この結果はロジットの 1001 要素の配列で、画像がそれぞれのクラスである確率に基づく数値が割り振られ、順位付けされています。
#そのため、トップのクラス ID は argmax を使うことでみつけることができます:
predicted_class = np.argmax(result[0], axis=-1)
predicted_class

#推論結果のデコード
#ここまでの内容で推論結果のクラス ID を表す数値が得られるので、 ImageNet のラベルを取得し、推論結果をデコードします
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())

plt.imshow(grace_hopper)
plt.axis('off')
predicted_class_name = imagenet_labels[predicted_class]
_ = plt.title("Prediction: " + predicted_class_name.title())

#シンプルな転移学習
#TF Hub を利用することで、データセットのクラスを認識するための、モデルの最上位層を簡単に再訓練することができます。

#データセット
#こちらの例では、TensorFlow flowers データセットを利用して進めます:
data_root = tf.keras.utils.get_file(
  'flower_photos','https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
   untar=True)

#このデータをモデルにロードするには tf.keras.preprocessing.image.ImageDataGenerator を使うのがもっとも簡単な方法です。
#すべての TensorFlow Hub の画像モジュールは [0, 1] の範囲で float で入力されることを想定しています。入力をリスケールする際には ImageDataGenerator の rescale パラメータを利用してください。
#画像のサイズは後ほど処理されます。
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
image_data = image_generator.flow_from_directory(str(data_root), target_size=IMAGE_SHAPE)

#結果のオブジェクトは image_batch, label_batch のペアを返すイテレーターです。
for image_batch, label_batch in image_data:
  print("Image batch shape: ", image_batch.shape)
  print("Label batch shape: ", label_batch.shape)
  break

#分類器で画像をバッチ処理する
#分類器で画像をバッチ処理していきます。
result_batch = classifier.predict(image_batch)
print(result_batch.shape)

predicted_class_names = imagenet_labels[np.argmax(result_batch, axis=-1)]
print(predicted_class_names)

#次に、これらの推論結果が画像とどのように一致するかを確認します:
plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  plt.title(predicted_class_names[n])
  plt.axis('off')
_ = plt.suptitle("ImageNet predictions")

#画像の帰属については LICENSE.txt をご参照ください。
#完璧とは程遠い結果ですが、このモデルはこれらのクラスのために訓練されたものではない（ただし、"daisy" を除いて）ということを考慮すると、妥当な結果だといえるでしょう。



#ヘッドレスモデルのダウンロード
#TensorFlow Hub は最上位の分類層を含まないモデルも配布しています。これらは転移学習に簡単に利用することができます。
#tfhub.dev にある、いずれの TensorFlow 2 互換の image feature vector URL でも、こちらで動作します。
feature_extractor_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2" #@param {type:"string"}

#特徴抽出器 (feature extractor) を作成します。
feature_extractor_layer = hub.KerasLayer(feature_extractor_url,
                                         input_shape=(224,224,3))

#これは画像毎に長さ 1280 のベクトルデータを返します:
feature_batch = feature_extractor_layer(image_batch)
print(feature_batch.shape)

#特徴抽出レイヤーの変数をフリーズして、訓練が新しい分類器のレイヤーのみを変更するようにします。
feature_extractor_layer.trainable = False


#上位の分類層を接合する
#tf.keras.Sequential モデルの hub レイヤーをラップして、新しい分類層を追加します。
model = tf.keras.Sequential([
  feature_extractor_layer,
  layers.Dense(image_data.num_classes, activation='softmax')
])

model.summary()

predictions = model(image_batch)

predictions.shape


#モデルの訓練
#訓練プロセスのコンフィグにはコンパイルを使用します:
model.compile(
  optimizer=tf.keras.optimizers.Adam(),
  loss='categorical_crossentropy',
  metrics=['acc'])

#モデルの訓練には .fit メソッドを使用します。
#この例を短く保つために訓練は 2 エポックだけにします。訓練のプロセスを可視化するために、
##各エポックの平均だけではなく各々のバッチで個別に損失と正確度を記録するためのカスタムコールバックを使用します。
class CollectBatchStats(tf.keras.callbacks.Callback):
  def __init__(self):
    self.batch_losses = []
    self.batch_acc = []

  def on_train_batch_end(self, batch, logs=None):
    self.batch_losses.append(logs['loss'])
    self.batch_acc.append(logs['acc'])
    self.model.reset_metrics()

steps_per_epoch = np.ceil(image_data.samples/image_data.batch_size)

batch_stats_callback = CollectBatchStats()

history = model.fit_generator(image_data, epochs=2,
                              steps_per_epoch=steps_per_epoch,
                              callbacks = [batch_stats_callback])

#これで、ほんの数回の訓練の繰り返しでさえ、タスクにおいてモデルが進歩していることが分かりました。

plt.figure()
plt.ylabel("Loss")
plt.xlabel("Training Steps")
plt.ylim([0,2])
plt.plot(batch_stats_callback.batch_losses)


plt.figure()
plt.ylabel("Accuracy")
plt.xlabel("Training Steps")
plt.ylim([0,1])
plt.plot(batch_stats_callback.batch_acc)


#推論結果の確認
#前からプロットをやり直すには、まずクラス名のリストを取得します:

class_names = sorted(image_data.class_indices.items(), key=lambda pair:pair[1])
class_names = np.array([key.title() for key, value in class_names])
print(class_names)

#画像のバッチをモデルに入力し、得られた ID をクラス名に変換します。
predicted_batch = model.predict(image_batch)
predicted_id = np.argmax(predicted_batch, axis=-1)
predicted_label_batch = class_names[predicted_id]

#結果をプロットします
label_id = np.argmax(label_batch, axis=-1)

plt.figure(figsize=(10,9))
plt.subplots_adjust(hspace=0.5)
for n in range(30):
  plt.subplot(6,5,n+1)
  plt.imshow(image_batch[n])
  color = "green" if predicted_id[n] == label_id[n] else "red"
  plt.title(predicted_label_batch[n].title(), color=color)
  plt.axis('off')
_ = plt.suptitle("Model predictions (green: correct, red: incorrect)")


#モデルのエクスポート
#訓練が完了したので、saved model としてエクスポートします:
import time
t = time.time()
print('----------------------------------------------------------------------------')
export_path = "/saved_models/{}".format(int(t))
model.save(export_path, save_format='tf')

export_path

#リロードできること、そしてエクスポートする前とおなじ結果が得られることを確認します:
reloaded = tf.keras.models.load_model(export_path)

result_batch = model.predict(image_batch)
reloaded_result_batch = reloaded.predict(image_batch)

abs(reloaded_result_batch - result_batch).max()

#この saved model は後々推論を行うため、そして TFLite や TFjs のモデルに変換するためにロードできます。