#!/bin/bash

input_file=""
output_dir=""
input_name=""
docker_args=""

# Function to modify --out argument
modify_dir_argument() {
  local args=("$@")
  local index=0
  echo "Replacing arguments for docker run">&2

  for ((i=0; i<${#args[@]}; i++)); do
    echo "inspecting ${args[i]} ${i}/${#args[@]}">&2
    if [[ "${args[i]}" == "--out" ]]; then
      index=$((i + 1))
      declare -g output_dir="${args[index]}"
      echo "output_dir set to ${output_dir}">&2
      # Map the original DIR to the desired directory
      args[index]="/data/out/"
      echo "Replaced '--out ${output_dir}' -> '--out ${args[index]}'">&2
    fi
    if [[ "${args[i]}" == "--input" ]]; then
      index=$((i + 1))
      declare -g input_file="${args[index]}"
      echo "input_file set to ${input_file}">&2
      # Map the original input to the desired file
      mapped_file="/data/out/$dir"
      args[index]="$mapped_dir"
      # Extract the file name and extension
      file_name=$(basename -- "$input_file")
      file_extension="${file_name##*.}"
      file_name="${file_name%.*}"
      input_name="input.${file_extension}"
      args[index]="/data/${input_name}"
      args[i]="-v"
      echo "Replaced '--input ${input_file}' -> '--input ${args[index]}'">&2
    fi
  done

  declare -g docker_args=("${args[@]}")
}

echo "checking arguments..."
modify_dir_argument "$@"
echo "checking arguments... done"
      echo "output_dir set to ${output_dir}">&2


# Check if the path is absolute
if [[ ! "$output_dir" = /* ]]; then
    # Make the path absolute
    output_dir=$(realpath "$output_dir")

    if [ $? -ne 0 ]; then
        echo "Error: Unable to convert the path to absolute."
        exit 1
    fi

    echo "Converted to absolute path: $output_dir"
fi
mkdir -p "${output_dir}"

# Check if the path is absolute
if [[ ! "$input_file" = /* ]]; then
    # Make the path absolute
    input_file=$(realpath "$input_file")

    if [ $? -ne 0 ]; then
        echo "Error: Unable to convert the path to absolute."
        exit 1
    fi

    echo "Converted to absolute path: $input_file"
fi

mkdir -p "${output_dir}"
echo "docker run --network none --shm-size 8G --gpus all -v '${output_dir}:/data/out/' -v '${input_file}:/data/${input_name}' verbatim verbatim ${docker_args[@]} -o /data/out"
docker run --network none --shm-size 8G --gpus all -v "${output_dir}:/data/out/" -v "${input_file}:/data/${input_name}" verbatim verbatim ${docker_args[@]} -o /data/out

