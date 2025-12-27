FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /project

# Install system dependencies
COPY .agent/ai-workbench/apt.txt /tmp/apt.txt
RUN apt-get update && xargs -a /tmp/apt.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Post-build steps
COPY .agent/ai-workbench/postBuild.bash /tmp/postBuild.bash
RUN bash /tmp/postBuild.bash

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
