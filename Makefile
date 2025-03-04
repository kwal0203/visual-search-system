clean:
	rm -f src/storage_service/mnist.db
	rm -f src/embedding_service/model/embedding_model.pth
	find src/storage_service/processed -name "*.png" -delete
	rm -f src/embedding_service/results/loss.png
	rmdir src/embedding_service/results