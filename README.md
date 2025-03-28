# Inference Perf

The Inference Perf project aims to provide GenAI inference performance benchmarking tool. It came out of [wg-serving](https://github.com/kubernetes/community/tree/master/wg-serving) and is sponsored by [SIG Scalability](https://github.com/kubernetes/community/blob/master/sig-scalability/README.md#inference-perf). See the [proposal](https://github.com/kubernetes-sigs/wg-serving/tree/main/proposals/013-inference-perf) for more info.

## Status

This project is currently in development.

## Getting Started

[PDM Python Package Manager](https://pdm-project.org/latest/) is utilized in this repository for dependecy management.

- Setup virtual environment with `pdm` and install dependencies

    ```
    make all-deps
    ```

- Run inference-perf CLI

    ```
    pdm run inference-perf --config_file config.yml
    ```

## Contributing

Our community meeting is weekly at Th 11:30 PDT ([Zoom Link](https://zoom.us/j/9955436256?pwd=Z2FQWU1jeDZkVC9RRTN4TlZyZTBHZz09), [Meeting Notes](https://docs.google.com/document/d/15XSF8q4DShcXIiExDfyiXxAYQslCmOmO2ARSJErVTak/edit?usp=sharing)).

We currently utilize the [#wg-serving](https://kubernetes.slack.com/?redir=%2Fmessages%2Fwg-serving) Slack channel for communications.

Contributions are welcomed, thanks for joining us!

### Code of conduct

Participation in the Kubernetes community is governed by the [Kubernetes Code of Conduct](code-of-conduct.md).
