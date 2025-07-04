# Evaluating RAG answer safety

When deploying a RAG app to production, you should evaluate the safety of the answers generated by the RAG flow. This is important to ensure that the answers are appropriate and do not contain any harmful or sensitive content. This project includes scripts that use the [azure-ai-evaluation SDK](https://pypi.org/project/azure-ai-evaluation/#history) to perform an [automated safety scan with an AI Red Teaming agent](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent).

* [Deploy an Azure AI project](#deploy-an-azure-ai-project)
* [Setup the evaluation environment](#setup-the-evaluation-environment)
* [Run red teaming agent](#run-red-teaming-agent)
* [Review the red teaming results](#review-the-red-teaming-results)

## Deploy an Azure AI project

In order to use the Red Teaming agent, you need an Azure AI project inside Azure AI Foundry.

1. Run this command to tell `azd` to provision an Azure AI project:

    ```shell
    azd env set USE_AI_PROJECT true
    ```

2. Then, run the following command to provision the project:

    ```shell
    azd provision
    ```

## Setup the evaluation environment

1. Create a new Python virtual environment in `.evalenv` by running the following command:

    ```bash
    python -m venv .evalenv
    ```

2. Activate the virtual environment by running the following command:

    MacOS/Linux:

    ```bash
    source .evalenv/bin/activate
    ```

    Windows:

    ```bash
    .evalenv\Scripts\activate
    ```

3. Install the dependencies for the safety evaluation script:

  ```bash
  pip install uv
  uv pip install -r evals/requirements.txt
  ```

## Run the red teaming agent

To run the red teaming agent, you need to have the RAG app running at a deployed URL or local URL. Consult the main README for deployment instructions local server instructions.

In the same terminal where you activated `.evalenv`, run this command to perform a red teaming scan:

```shell
python evals/safety_evaluation.py --target_url <TARGET_URL> --questions_per_category <NUM_QUESTIONS>
```

* `--target_url`: The target URL for the callback. Default is `http://127.0.0.1:8000/chat`, which assumes the RAG app is running locally. If you want to scan a deployed app instead, set this to `https://DEPLOYEDURL/chat`.
* `--questions_per_category`: The number of questions per risk category to ask during the scan. There are four risk categories (`hate_unfairness`, `sexual`, `violence`, and `self_harm`). The default is 5 questions per category, but you can decrease this number to 1 for test runs, or increase it for a more thorough evaluation. Note that the total questions asked will be higher, as the red teaming agent is configured to try [multiple attack strategies](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent#supported-attack-strategies), where it transforms the original question in different ways.

## Review the red teaming results

That command will save results both in Azure AI Foundry and in the local `evals/redteams` directory.

### Review results in Azure AI Foundry

Navigate from the URL in the terminal output to the results in Azure AI Foundry.
First you'll see overall attack metrics:

![Red Teaming Agent Results](images/redteam_dashboard.png)

Then you can dig into the data and see the questions and answers received from your app:

![Red Teaming Agent Results](images/redteam_logs.png)

Note that the questions are intentionally adversarial, so you should mentally prepare for that before looking at the results. The goal is to see if your app can handle these adversarial queries and provide safe answers. Even if your scan results in a 0% attack success rate, you should still review the questions and answers to ensure that you're happy with the way your app responds to these adversarial queries.

Learn more in the [Red Teaming Agent documentation](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent#viewing-your-results-in-azure-ai-foundry-project).

## Review results locally

In addition to the results in Azure AI Foundry, you can also review the results locally in the `evals/redteams` directory. The results are saved in JSON format.

Each file in this directory corresponds to a single red teaming scan, and contains the following fields:

* `scorecard`: A summary of the scan results, including the attack success rate and the number of questions asked.
* `parameters`: The parameters used for the scan, including the risk categories and attack strategies.
* `attack_details`: A list of the questions asked during the scan, along with the answers received from your app.
* `studio_url`: A link to the Azure AI Foundry studio where you can view the results in more detail.

Learn more in the [Red Teaming Agent documentation](https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent#results-from-your-automated-scans).

## Resources

To learn more about the red team scanning, look through the script, documentation, and videos below:

* [safety_evaluation.py](/evals/safety_evaluation.py)
* [Run automated safety scans with AI Red Teaming Agent](https://learn.microsoft.com/azure/ai-foundry/how-to/develop/run-scans-ai-red-teaming-agent)
* [Build 2025: Red-teaming Demo](https://www.youtube.com/watch?v=sZzcSX7BFVA)
