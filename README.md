# riwwer_ki_demo_frontend

## Deployment
- Build docker image
```bash
docker buildx build --platform linux/amd64 -t registry.datexis.com/calgo-lab/riwwer-ki-demo .
```

- Push docker image
```bash
docker push registry.datexis.com/calgo-lab/riwwer-ki-demo
```

- Deploy pod
```bash
kubectl delete deploy riwwer-ki-demo -n calgo-lab
kubectl apply -f k8s/deployment.yaml -n calgo-lab
```