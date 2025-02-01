class FireDetection:
    @staticmethod
    def is_fire_detected(results):
        # Replace with actual logic to determine if fire is detected
        for result in results.xyxy[0]:
            if result[-1] == 'fire':  # Assuming 'fire' is the label for fire in the model
                return True
        return False