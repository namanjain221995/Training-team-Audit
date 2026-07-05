# Lambda patch — wake the worker when a job is enqueued

The worker instance stops itself when idle. The existing
`zoom-recording-processor` Lambda must START it whenever it enqueues a job.
`start_instances` on an already-running instance is a no-op, so call it every time.

## 1. Code (add to lambda_function.py)

```python
WORKER_INSTANCE_ID = os.environ.get("WORKER_INSTANCE_ID", "").strip()

def wake_worker():
    """Start the analysis EC2 worker (idempotent). Failure is non-fatal:
    the job waits in SQS (retention 4 days) until the worker is up."""
    if not WORKER_INSTANCE_ID:
        return
    try:
        boto3.client("ec2").start_instances(InstanceIds=[WORKER_INSTANCE_ID])
        print(f"worker start requested: {WORKER_INSTANCE_ID}")
    except Exception as exc:
        print(f"could not start worker ({exc}); job stays queued")
```

Call `wake_worker()` right after the successful
`sqs.send_message(...)` in `enqueue_analysis_job` (immediately before or after
flipping `enqueued: true` in training-temp.json).

## 2. Lambda environment

    WORKER_INSTANCE_ID = i-xxxxxxxxxxxxxxxxx

## 3. Lambda IAM (add statement)

```json
{
  "Effect": "Allow",
  "Action": "ec2:StartInstances",
  "Resource": "arn:aws:ec2:us-east-1:985100584614:instance/i-xxxxxxxxxxxxxxxxx"
}
```

## Flow after this patch

    temp json written -> job enqueued -> wake_worker()
      instance stopped  -> boots, self-updates, drains queue
      instance running  -> no-op, worker takes the job next
      after IDLE_MINUTES with an empty queue -> worker stops the instance
