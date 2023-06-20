import random

def generate_points(num_points, x_range, y_range):
    """주어진 범위 내에서 무작위로 점을 생성하는 함수"""
    points = []
    for _ in range(num_points):
        x = random.randint(x_range[0], x_range[1])
        y = random.randint(y_range[0], y_range[1])
        points.append((x, y))
    return points

def classify_points(points, centroids):
    """주어진 점들을 클러스터 중심값에 가까운 점들로 분류하는 함수"""
    clusters = {}
    for point in points:
        min_dist = float('inf')
        closest_centroid = None
        for centroid in centroids:
            dist = ((point[0] - centroid[0]) ** 2 + (point[1] - centroid[1]) ** 2) ** 0.5
            if dist < min_dist:
                min_dist = dist
                closest_centroid = centroid
        if closest_centroid in clusters:
            clusters[closest_centroid].append(point)
        else:
            clusters[closest_centroid] = [point]
    return clusters

# 사용자로부터 클러스터 중심 개수 입력 받기
num_centroids = int(input("클러스터 중심 개수를 입력하세요: "))

# x축과 y축의 범위
x_range = (-100, 100)
y_range = (-100, 100)

# N개의 점 생성
points = generate_points(1000, x_range, y_range)

# 클러스터 중심값 생성
centroids = []
for _ in range(num_centroids):
    centroid = (random.randint(x_range[0], x_range[1]), random.randint(y_range[0], y_range[1]))
    centroids.append(centroid)

# 점들을 클러스터 중심값에 분류
clusters = classify_points(points, centroids)

# 결과를 result.txt 파일로 저장
with open("result.txt", "w") as f:
    for centroid, cluster_points in clusters.items():
        data = [centroid, cluster_points]
        f.write(str(data) + "\n")

# centroids 리스트에 있는 각 클러스터 중심값을 리스트 형식으로 변환
centroid_list = [eval(str(centroid)) for centroid in centroids]

with open('c_result.txt', 'w') as f:
    for i, centroid in enumerate(centroid_list):
        f.write(f"{i}, {centroid[0]}, {centroid[1]}\n")


# 클러스터 중심값과 클러스터에 속하는 점들 출력
#for line in lines:
#    line = line.strip()
#    parts = line.split(", ")
#    centroid = parts[0]
#    points = parts[1:]

#    print("Centroid:", centroid)
#    print("Points in cluster:", points)

# 결과 출력
#print("클러스터 중심값: ", centroids)
#for centroid, cluster_points in clusters.items():
#    print(f"중심값 {centroid}에 속하는 점들: {cluster_points}")

