pipeline {
    agent any

    triggers {
        githubPush()
    }

    options {
        timestamps()
        timeout(time: 10, unit: 'MINUTES')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }

        stage('Test in Docker') {
            steps {
                script {
                    docker.image('python:3.11-slim').inside {
                        sh '''
                            pip install --disable-pip-version-check --no-cache-dir -r requirements.txt pytest httpx
                            pytest -q
                        '''
                    }
                }
            }
        }

        stage('Deploy Part II with Docker Compose') {
            steps {
                sh '''
                    test -r /etc/qaibridge/assignment2.env
                    docker compose \
                      --env-file /etc/qaibridge/assignment2.env \
                      -f docker-compose-part2.yml \
                      up -d --force-recreate

                    for attempt in $(seq 1 36); do
                      status=$(docker inspect --format='{{.State.Health.Status}}' qaibridge-part2-app 2>/dev/null || true)
                      if [ "$status" = healthy ]; then
                        echo "Part II application is healthy."
                        exit 0
                      fi
                      sleep 5
                    done

                    docker compose \
                      --env-file /etc/qaibridge/assignment2.env \
                      -f docker-compose-part2.yml logs
                    exit 1
                '''
            }
        }
    }

    post {
        always {
            script {
                def commitEmail = sh(
                    script: 'git log -1 --format=%ae || true',
                    returnStdout: true
                ).trim()
                if (commitEmail) {
                    emailext(
                        to: commitEmail,
                        subject: "Qaibridge Assignment 2 build ${currentBuild.currentResult}: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                        body: """Build result: ${currentBuild.currentResult}
Job: ${env.JOB_NAME}
Build: ${env.BUILD_URL}
"""
                    )
                } else {
                    echo 'No commit author email was available; build email was not sent.'
                }
            }
        }
    }
}
