from datetime import datetime, UTC

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    age = Column(Integer, nullable=True)
    gender = Column(String, nullable=True)
    city = Column(String, nullable=True)
    country = Column(String, nullable=True)
    email = Column(String, unique=True, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # Relationships
    images = relationship("Image", back_populates="user")
    interactions = relationship("UserImageInteraction", back_populates="user")


class Image(Base):
    __tablename__ = "images"

    image_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    file_path = Column(String)
    embedding = Column(String, nullable=True)  # JSON string of embedding vector
    tags = Column(String)  # JSON string of tags
    created_at = Column(DateTime, default=lambda: datetime.now(UTC))

    # MNIST specific fields
    digit_label = Column(Integer, nullable=True)  # The actual digit (0-9)
    is_mnist = Column(Boolean, default=False)  # Flag to identify MNIST images
    dataset_split = Column(String, nullable=True)  # 'train' or 'test'

    # Relationships
    user = relationship("User", back_populates="images")
    query_interactions = relationship(
        "UserImageInteraction",
        foreign_keys="UserImageInteraction.query_image_id",
        back_populates="query_image",
    )
    display_interactions = relationship(
        "UserImageInteraction",
        foreign_keys="UserImageInteraction.displayed_image_id",
        back_populates="displayed_image",
    )


class UserImageInteraction(Base):
    __tablename__ = "user_image_interactions"

    interaction_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"))
    query_image_id = Column(Integer, ForeignKey("images.image_id"))
    displayed_image_id = Column(Integer, ForeignKey("images.image_id"))
    position_in_ranked_list = Column(Integer)
    click = Column(Boolean, default=False)
    interaction_time = Column(DateTime, default=lambda: datetime.now(UTC))

    # Additional metrics for evaluation
    time_to_click = Column(
        Float, nullable=True
    )  # Time in seconds from display to click
    session_id = Column(String, index=True)  # To group interactions in the same session

    # Relationships
    user = relationship("User", back_populates="interactions")
    query_image = relationship(
        "Image", foreign_keys=[query_image_id], back_populates="query_interactions"
    )
    displayed_image = relationship(
        "Image",
        foreign_keys=[displayed_image_id],
        back_populates="display_interactions",
    )
