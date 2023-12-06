import boto3
import io
import s3fs
import smart_open
import json
import lib
import whisper

import soundfile as sf
from pathlib import Path
from pyspark.sql import Row

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, FloatType, ArrayType

bucket_name = "data-engineer-test"
file_path='output/folder_name=p225/data.parquet'
flac_object_key = "wav48_silence_trimmed/p225/p225_001_mic1.flac"

# Opening JSON file
f = open('config.json')

# returns JSON object as
# a dictionary
config = json.load(f)


def spark_data_pipeline(output_file='result.parquet'):
    s3_client = boto3.client(
        service_name ="s3",
        endpoint_url = config['endpoint_url'],
        aws_access_key_id = config['aws_access_key_id'],
        aws_secret_access_key = config['aws_secret_access_key'],
    )

    # List all keys in the bucket
    response = s3_client.list_objects(Bucket=bucket_name)

    flac_files = [elt['Key'] for elt in response['Contents'] if elt['Key'].endswith(".flac")]
    model = whisper.load_model("base")

    # Create a Spark session
    spark = SparkSession.builder.appName("metavoice").getOrCreate()

    # Define the schema for the empty DataFrame
    schema = StructType([
        StructField("id", StringType(), True),
        StructField("transcription", StringType(), True),
        StructField("token_array", ArrayType(FloatType()), True)
    ])

    df = spark.createDataFrame([], schema)

    # Set the duration of each chunk in seconds
    chunk_duration = 300

    for file_nb, flac_object_key in enumerate(flac_files):
        s3_uri = f's3://{bucket_name}/{flac_object_key}'
        print(s3_uri)
        # Open the FLAC audio file from S3 using smart_open
        with smart_open.open(uri=s3_uri, mode='rb', transport_params={'client': s3_client}) as s3_file:
            # Initialize variables
            current_time = 0
            i = 0
            transcriptions = []
            tokenised_audio = []
            # Iterate through the audio file in chunks
            while True:
                i += 1
                # Read a chunk of FLAC data
                flac_chunk = s3_file.read(chunk_duration * 1000)  # Read in milliseconds

                # Break if the end of the file is reached
                if not flac_chunk:
                    sentence = ' '.join(transcriptions)
                    flac_file_path = Path(flac_object_key)
                    wav_file_path = str(flac_file_path.with_suffix(".wav"))
                    tokenised_audio = [float(x) for x in tokenised_audio]
                    new_record = Row(id=wav_file_path, transciption=sentence,
                                     token_array=tokenised_audio)
                    df = df.union(spark.createDataFrame([new_record], schema=schema))
                    #df.show()
                    break

                wav_data = lib.convert_flac_to_wav(flac_chunk)

                # Open the WAV data using soundfile
                with io.BytesIO(wav_data) as wav_io:
                    with sf.SoundFile(wav_io, 'r') as audio_file:
                        audio_chunk = audio_file.read(dtype='float32')
                        result = model.transcribe(audio_chunk)
                        tokenised_audio.extend(lib.tokenise(audio_chunk))
                        transcriptions.append(result['text'])

                        # Print the current time
                        print(f"Current Time: {current_time:.2f}s")

                        # Update the current time for the next iteration
                        current_time += chunk_duration
    df.write.parquet(output_file)


if __name__ == '__main__':
    spark_data_pipeline()