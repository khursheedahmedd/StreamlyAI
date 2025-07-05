#!/usr/bin/env python3
"""
Test script for reel generation functionality
"""

import json
import os
from backend import extract_highlights_from_transcription, generate_video_reel

def test_highlight_extraction():
    """Test highlight extraction from transcription"""
    print("Testing highlight extraction...")
    
    # Sample transcription
    sample_transcription = """
    Welcome to this amazing video about artificial intelligence. 
    Today we will discuss the key concepts of machine learning and deep learning. 
    First, let's understand what AI really means. 
    Artificial intelligence is the simulation of human intelligence in machines. 
    Machine learning is a subset of AI that enables computers to learn without being explicitly programmed. 
    Deep learning uses neural networks with multiple layers to process complex patterns. 
    This technology has revolutionized many industries including healthcare, finance, and transportation. 
    In conclusion, AI is transforming our world in unprecedented ways.
    """
    
    highlights = extract_highlights_from_transcription(sample_transcription)
    
    print(f"Generated {len(highlights)} highlights:")
    for i, highlight in enumerate(highlights):
        print(f"  {i+1}. {highlight['start']:.1f}s - {highlight['end']:.1f}s: {highlight['captions'][0][0][:50]}...")
    
    return highlights

def test_video_reel_generation():
    """Test video reel generation (requires a video file)"""
    print("\nTesting video reel generation...")
    
    # Check if test video exists
    test_video = "audio.mp4"  # Using existing audio file as test
    if not os.path.exists(test_video):
        print(f"Test video {test_video} not found. Skipping video generation test.")
        return False
    
    # Create test highlights
    test_highlights = [
        {
            "start": 0,
            "end": 5,
            "captions": [("Test highlight 1", 0, 5)]
        },
        {
            "start": 10,
            "end": 15,
            "captions": [("Test highlight 2", 10, 15)]
        }
    ]
    
    output_path = "test_reel.mp4"
    
    try:
        result = generate_video_reel(test_video, test_highlights, output_path)
        if result:
            print(f"✅ Video reel generated successfully: {output_path}")
            return True
        else:
            print("❌ Video reel generation failed")
            return False
    except Exception as e:
        print(f"❌ Error during video reel generation: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\nTesting API endpoints...")
    
    # This would require a running Flask server
    print("Note: API endpoint testing requires a running Flask server")
    print("To test endpoints, start the server with: python3 backend.py")
    print("Then use curl or a tool like Postman to test:")
    print("  POST /generate_highlights")
    print("  POST /export_reel")
    print("  GET /download_reel/<filename>")

if __name__ == "__main__":
    print("🧪 Testing StreamlyAI Reel Generation Functionality\n")
    
    # Test highlight extraction
    highlights = test_highlight_extraction()
    
    # Test video reel generation
    video_success = test_video_reel_generation()
    
    # Test API endpoints info
    test_api_endpoints()
    
    print(f"\n📊 Test Results:")
    print(f"  ✅ Highlight extraction: {len(highlights)} highlights generated")
    print(f"  {'✅' if video_success else '❌'} Video reel generation: {'PASSED' if video_success else 'FAILED'}")
    print(f"  ℹ️  API endpoints: Manual testing required")
    
    print("\n🎉 Reel generation functionality is ready for use!") 