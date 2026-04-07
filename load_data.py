from pathlib import Path as path
cwd = path.cwd()
DATA_DIR = rf"{cwd}\data"

sample_data = {
    rf"{DATA_DIR}\ec2.txt" : """
Amazon EC2 (Elastic Compute Cloud) is a core Infrastructure-as-a-Service (IaaS) offering from AWS that provides resizable compute capacity in the cloud through virtual machines known as instances. Its primary purpose is to eliminate the need for organizations to invest in and maintain physical servers, enabling rapid provisioning, scalability, and flexibility in running applications. EC2 supports a wide range of workloads including web hosting, enterprise applications, big data processing, machine learning, and high-performance computing.

EC2 instances are highly configurable, allowing users to choose from a variety of Amazon Machine Images (AMIs), which are preconfigured templates containing operating systems (such as Linux or Windows) and optional application software. Users can select instance types categorized into families such as General Purpose (balanced resources), Compute Optimized (high CPU performance), Memory Optimized (large RAM capacity), Storage Optimized (high disk throughput), and Accelerated Computing (GPU-based workloads). Each instance type varies in CPU, memory, storage, and networking capabilities.

Networking in EC2 is managed through Amazon Virtual Private Cloud (VPC), allowing users to launch instances within isolated virtual networks, configure IP addressing, subnets, route tables, and gateways. Security is enforced using Security Groups (stateful firewalls controlling inbound/outbound traffic) and Network ACLs (stateless filtering at the subnet level). EC2 also supports Elastic IP addresses, which are static public IPv4 addresses that can be remapped between instances.

Storage options include Amazon Elastic Block Store (EBS) for persistent block storage volumes, instance store (ephemeral storage physically attached to the host machine), and integration with object storage via Amazon S3. EBS volumes support different performance tiers such as SSD-backed volumes for low latency and HDD-backed volumes for throughput-intensive workloads. Snapshots can be taken for backup and disaster recovery.

EC2 provides high availability and fault tolerance through features like Availability Zones (isolated data centers within a region), Auto Scaling groups (which automatically adjust the number of instances based on demand or health checks), and Elastic Load Balancing (which distributes incoming traffic across multiple instances). Monitoring is handled through Amazon CloudWatch, which collects metrics, logs, and triggers alarms.

Additional features include placement groups for optimizing instance placement (cluster, spread, partition strategies), hibernation for pausing and resuming instances, and dedicated hosts or instances for compliance requirements. EC2 also integrates with IAM (Identity and Access Management) for secure access control using roles and policies.

The pricing model for EC2 is flexible and includes several options: On-Demand pricing (pay per use with no commitment), Reserved Instances (long-term commitments with significant discounts), Savings Plans (flexible pricing based on usage commitment), and Spot Instances (deep discounts for unused capacity with the possibility of interruption). Costs are influenced by instance type, operating system, region, usage duration, storage, and data transfer. Additional charges may apply for premium features such as detailed monitoring or advanced networking.
    """,

     rf"{DATA_DIR}\s3.txt" : """
   Amazon S3 (Simple Storage Service) is a fully managed object storage service designed for scalability, durability, availability, and security. Its primary purpose is to store and retrieve any amount of data from anywhere on the web, making it suitable for a wide variety of use cases including backup and restore, archival storage, content distribution, big data analytics, and application data storage.

S3 organizes data into buckets, which act as containers for objects. Each object consists of the data itself, metadata (such as content type and timestamps), and a unique key used for retrieval. Buckets are globally unique and can be configured within specific AWS regions. S3 provides strong read-after-write consistency for new objects and eventual consistency for overwrite and delete operations.

One of the defining characteristics of S3 is its durability, designed for 99.999999999% (11 nines) durability by automatically replicating data across multiple devices and facilities within a region. It also offers high availability and supports versioning, which allows users to preserve, retrieve, and restore every version of an object, protecting against accidental deletion or overwrites.

S3 offers multiple storage classes tailored for different access patterns and cost requirements. These include S3 Standard (frequent access), S3 Intelligent-Tiering (automatic cost optimization), S3 Standard-IA (infrequent access), S3 One Zone-IA (lower cost, single AZ), and S3 Glacier tiers (Instant Retrieval, Flexible Retrieval, Deep Archive) for long-term archival. Lifecycle policies enable automatic transitions between these classes or deletion of objects based on predefined rules.

Security in S3 is robust and includes features such as bucket policies, IAM policies, Access Control Lists (ACLs), and encryption options. Data can be encrypted at rest using server-side encryption (SSE-S3, SSE-KMS, SSE-C) or client-side encryption. Data in transit is secured using HTTPS. S3 also supports features like Block Public Access to prevent accidental exposure of sensitive data.

S3 integrates with many AWS services, including CloudFront for content delivery, Lambda for event-driven processing, and Athena for querying data directly in S3 using SQL. Event notifications allow triggering workflows when objects are created, deleted, or modified.

The pricing model for S3 is based on storage usage (per GB per month), request frequency (PUT, GET, LIST, etc.), data transfer out, and additional features such as replication or data retrieval from archival tiers. Costs vary significantly depending on the chosen storage class, with colder storage tiers offering lower storage costs but higher retrieval latency and fees.
""" ,

 rf"{DATA_DIR}\lambda.txt" : """
AWS Lambda is a serverless compute service that allows developers to run code in response to events without provisioning or managing servers. Its primary purpose is to abstract away infrastructure management, enabling developers to focus solely on writing and deploying code while AWS handles scaling, availability, and execution.

Lambda operates on an event-driven model, where functions are triggered by events from various AWS services such as S3 (file uploads), API Gateway (HTTP requests), DynamoDB (database changes), CloudWatch (scheduled events), and more. Each function runs in a stateless execution environment, meaning no data is retained between executions unless stored externally.

Lambda supports multiple programming languages, including Python, Node.js, Java, Go, Ruby, and .NET, and allows custom runtimes. Functions are deployed as packages containing code and dependencies, and configuration includes memory allocation, timeout settings, and environment variables. Memory allocation directly impacts CPU and network performance.

A key feature of Lambda is automatic scaling, where the service can handle a few requests per day or thousands per second without manual intervention. Concurrency controls allow limiting or reserving execution capacity. Lambda ensures high availability by running functions across multiple availability zones.

Lambda integrates deeply with the AWS ecosystem, enabling microservices architectures, real-time file processing, stream processing, and backend APIs. It can be combined with API Gateway to create fully serverless web applications or used with Step Functions for orchestrating workflows.

Advanced features include Lambda Layers (for sharing code across functions), provisioned concurrency (to reduce cold start latency), and support for container images (up to 10 GB). Monitoring and logging are handled through CloudWatch Logs and metrics, while AWS X-Ray provides distributed tracing for debugging and performance analysis.

Security is managed through IAM roles assigned to Lambda functions, ensuring least-privilege access to other AWS resources. Lambda also runs code in isolated environments for enhanced security.

The pricing model is based on the number of requests and execution duration (measured in milliseconds), as well as the amount of memory allocated. Additional charges may apply for provisioned concurrency. A free tier is available, offering a baseline number of requests and compute time per month, making Lambda highly cost-effective for intermittent or unpredictable workloads.
"""
}


for file_path, service_data in sample_data.items():
    with open(file_path,'w',encoding='utf-8') as f:
        f.write(service_data)
        print("file created....")